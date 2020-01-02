#pragma once
//------------------------------------------------------------------------------
//
//   Copyright 2018-2020 Fetch.AI Limited
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//------------------------------------------------------------------------------

#include "logging/logging.hpp"
#include "ml/ops/ops.hpp"
#include "ml/ops/weights.hpp"
#include "ml/saveparams/saveable_params.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fetch {
namespace ml {

template <typename TensorType>
class Node
{
private:
  enum class CachedOutputState : uint8_t
  {
    VALID_CACHE,
    CHANGED_CONTENT,
    CHANGED_SIZE
  };

public:
  using DataType        = typename TensorType::Type;
  using NodeWeakPtrType = std::weak_ptr<Node<TensorType>>;

  using VecTensorType    = typename fetch::ml::ops::Ops<TensorType>::VecTensorType;
  using SPType           = fetch::ml::NodeSaveableParams<TensorType>;
  using NodeErrorMapType = std::unordered_map<Node<TensorType> *, std::vector<TensorType>>;
  using Shape            = fetch::math::SizeVector;
  using ShapeVector      = std::vector<Shape>;

  ///////////////////////////////////
  /// CONSRTUCTORS / DESCTRUCTORS ///
  ///////////////////////////////////

  Node() = default;

  Node(OpType const &operation_type, std::string name,
       std::function<std::shared_ptr<ops::Ops<TensorType>>()> const &constructor)
    : name_(std::move(name))
    , cached_output_status_(CachedOutputState::CHANGED_SIZE)
    , operation_type_(operation_type)
  {
    op_ptr_ = constructor();
  }

  Node(OpType const &operation_type, std::string name, std::shared_ptr<ops::Ops<TensorType>> op_ptr)
    : name_(std::move(name))
    , cached_output_status_(CachedOutputState::CHANGED_SIZE)
    , operation_type_(operation_type)
    , op_ptr_(std::move(op_ptr))
  {}

  /**
   * This is used to make a copy of one node from another, i.e. when sharing weights.
   * @param old_node
   * @param name
   * @param op_ptr
   */
  Node(Node &old_node, std::string name, std::shared_ptr<ops::Ops<TensorType>> op_ptr)
    : name_(std::move(name))
    , cached_output_status_(CachedOutputState::CHANGED_SIZE)
    , operation_type_(old_node.OpCode())
    , op_ptr_(std::move(op_ptr))
  {
    cached_output_ = old_node.cached_output_.Copy();
  }

  virtual ~Node() = default;

  ///////////////////////
  /// SAVEABLE PARAMS ///
  ///////////////////////

  std::shared_ptr<SPType> GetNodeSaveableParams() const;

  void SetNodeSaveableParams(NodeSaveableParams<TensorType> const &nsp,
                             std::shared_ptr<ops::Ops<TensorType>> op_ptr);

  ///////////////////////////////////
  /// FORWARD/BACKWARD OPERATIONS ///
  ///////////////////////////////////

  VecTensorType               GatherInputs() const;
  std::shared_ptr<TensorType> Evaluate(bool is_training);

  NodeErrorMapType BackPropagate(TensorType const &error_signal);

  void                                AddInput(NodeWeakPtrType const &i);
  std::vector<std::string>            GetInputNames();
  void                                AddOutput(NodeWeakPtrType const &o);
  std::vector<NodeWeakPtrType> const &GetOutputs() const;
  void                                ResetCache(bool input_size_changed);
  void                                ResetInputsAndOutputs();

  std::string const &GetNodeName()
  {
    return name_;
  }

  std::shared_ptr<ops::Ops<TensorType>> GetOp()
  {
    return op_ptr_;
  }

  /**
   * returns the stored operation type and syncs it with operation type of
   * underlying Ops.
   * @return
   */
  OpType const &OpCode()
  {
    using namespace fetch::ml::string_helpers;
    if (operation_type_ != op_ptr_->OperationType())
    {
      FETCH_LOG_ERROR(this->name_.c_str(),
                      "Node operation type (" + OperationNames.at(operation_type_) +
                          ") and underlying Ops operation code (" +
                          OperationNames.at(op_ptr_->OperationType()) + ") mismatch!");
      operation_type_ = op_ptr_->OperationType();
    }
    return operation_type_;
  }

  bool HasValidCache()
  {
    return static_cast<bool>(cached_output_status_ == CachedOutputState::VALID_CACHE);
  }

  /**
   * @brief ForwardPassChargeCost calculates charge cost estimation of the current node.
   * The cost includes also charge costs for every preceding Node in a linear (sequential)
   * Graph up to input Node.
   * @return charge cost estimation for all Forward passes through all Nodes starting from input
   * and ending in this node.
   */
  // TODO(VH): Naming does not suggest recursive nature of this method, renaming needed.
  //           PrecedingNodesForwardCost or just PrecedingForwardCost?
  virtual fetch::vm::ChargeAmount ForwardPassChargeCost()
  {
    // TODO(VH): call OpForwardCost() of the Op - we need to know Data shape to do this.
    if (input_nodes_.empty())
    {
      auto const &expected_in_shapes = op_ptr_->ExpectedSliceInputShapes();
      if (this->OpCode() == OpType::OP_PLACEHOLDER)
      {
        FETCH_LOG_INFO(name_.c_str(), "Input Placeholder layer cost added");
      }
      else
      {
        FETCH_LOG_ERROR(name_.c_str(),
                        "Can not estimate a forward pass cost of a non-placeholder leaf Node!");
      }
      return op_ptr_->OpForwardCost(expected_in_shapes);
    }
    fetch::vm::ChargeAmount total_cost = 0;
    ShapeVector             input_shapes;
    for (auto const &i : input_nodes_)
    {
      if (auto input_node_ptr = i.lock())
      {
        total_cost += input_node_ptr->ForwardPassChargeCost();
        // TODO(VH): check if the calculations are valid.
        auto const in_shape = input_node_ptr->SliceOutputShape();
        if (!in_shape.empty())
        {
          input_shapes.emplace_back(in_shape);
        }
      }
      else
      {
        throw std::runtime_error("Unable to lock weak pointer.");
      }
    }
    auto const my_cost_of_input = op_ptr_->OpForwardCost(input_shapes);
    total_cost += my_cost_of_input;

    return total_cost;
  }

  void SetSliceOutputShape(Shape const &new_shape)
  {
    slice_output_shape_ = new_shape;
    op_ptr_->SetSliceOutputShape(new_shape);
  }

  /**
   * @brief SliceOutputShape computes an output shape of the Node, if only 1 data slice (e.g.
   * with batch size == 1) is provided to the Graph input. If there is no cached output shape,
   * the method is recursively called until either a cached shape or input Node is encountered.
   * @return vector of SizeType.
   */
  Shape SliceOutputShape()
  {
    // Returned cached shape result if available; also set underlying Op shape.
    if (!slice_output_shape_.empty())
    {
      if (op_ptr_->SliceOutputShape().empty())
      {
        op_ptr_->SetSliceOutputShape(slice_output_shape_);
        // TODO(VH): SetSliceInputShapes(...) ?
      }
      return slice_output_shape_;
    }

    // Is my Op pre-shaped?
    bool const i_am_preshaped =
        shaped_operations_.find(op_ptr_->OperationType()) != shaped_operations_.end();
    if (i_am_preshaped)
    {
      // TODO(VH): then what?..
      // Some Ops have their shape known at Graph compilation time
      Shape const candidate = op_ptr_->SliceOutputShape();
      if (candidate.empty())
      {
        // TODO(VH): throw error: shaped layer returned empty shape, linking failed!
        FETCH_LOG_ERROR(
            this->name_.c_str(),
            "A pre-shaped Node's underlying Op returned empty shape, shape linking impossible!");
        return slice_output_shape_;
      }
      slice_output_shape_ = candidate;
    }

    ShapeVector input_shapes;
    for (auto const &i : input_nodes_)
    {
      auto node_ptr = i.lock();
      if (!node_ptr)
      {
        throw std::runtime_error("Unable to lock weak pointer.");
      }
      // Deeper recursive call.
      auto const in_shape = node_ptr->SliceOutputShape();
      if (!in_shape.empty())
      {
        input_shapes.emplace_back(in_shape);
      }
      // TODO(VH): else (if there _is_ an empty shape among inputs) what?
    }

    if (!input_shapes.empty())
    {
      if (slice_output_shape_.empty())
      {
        slice_output_shape_ = op_ptr_->ComputeSliceOutputShape(input_shapes);
      }
      else
      {
        // TODO(VH): implement input shape matching validation.
      }
    }
    else
    {
      if (input_nodes_.size() != 1 || !i_am_preshaped)
      {
        FETCH_LOG_INFO(name_.c_str(), "Shape deduction reached a Graph leaf : " + this->name_);
        return slice_output_shape_;
      }
      // If I am a pre-shaped node, and my Input is a lone Placeholder, I can force its shape.
      std::shared_ptr<Node<TensorType>> input_node = input_nodes_.back().lock();
      if (!input_node)
      {
        throw std::runtime_error("Unable to lock weak pointer.");
      }
      if (input_node->OpCode() == OpType::OP_PLACEHOLDER)
      {
        input_node->SetSliceOutputShape(op_ptr_->ExpectedSliceInputShapes().front());
        FETCH_LOG_INFO(name_.c_str(), "Forced next layer shape to my expected input shape.");
      }
    }

    if (slice_output_shape_.empty())
    {
      // throw an error: invalid calcs from a previous layer.
      FETCH_LOG_INFO(name_.c_str(), "Error: shape deduction failed.");
    }
    return slice_output_shape_;
  }

private:
  std::vector<NodeWeakPtrType> input_nodes_;
  std::vector<NodeWeakPtrType> outputs_;

  std::string       name_;
  TensorType        cached_output_;
  CachedOutputState cached_output_status_;
  OpType operation_type_;  // TODO(VH): who synchronizes this code with underlying Ops OpCode?...

  // TODO(VH): only Ops should have computed Shapes, and the Node
  // must only ask its Op for output shape. "InformationExpert" violation.
  Shape slice_output_shape_{};

  std::shared_ptr<ops::Ops<TensorType>> op_ptr_;

  static const std::set<OpType> shaped_operations_;
};

/// A list of Ops that "knows" their output shape on construction, such as FullyConnected.
template <typename TensorType>
const std::set<OpType> Node<TensorType>::shaped_operations_{OpType::LAYER_FULLY_CONNECTED};

/**
 * Constructs and returns a NodeSaveableParams object allowing serialisation
 * @tparam T
 * @return
 */
template <typename TensorType>
std::shared_ptr<typename Node<TensorType>::SPType> Node<TensorType>::GetNodeSaveableParams() const
{
  auto sp_ptr = std::make_shared<SPType>();

  sp_ptr->name           = name_;
  sp_ptr->operation_type = operation_type_;
  sp_ptr->op_save_params = op_ptr_->GetOpSaveableParams();

  return sp_ptr;
}

/**
 * returns a vector of all nodes which provide input to this node
 * @tparam TensorType tensor
 * @return vector of reference_wrapped tensors
 */
template <class TensorType>
typename Node<TensorType>::VecTensorType Node<TensorType>::GatherInputs() const
{
  VecTensorType inputs;
  for (auto const &i : input_nodes_)
  {
    if (auto ptr = i.lock())
    {
      inputs.push_back(ptr->Evaluate(op_ptr_->IsTraining()));
    }
    else
    {
      throw std::runtime_error("Unable to lock weak pointer.");
    }
  }
  return inputs;
}

/**
 * Returns the result of a forward evaluation of this node. If that's already been
 * computed this is cheap; if not then Forward is called as necessary if the output
 * size has not been updated since last used. This also must be changed and
 * recalculated as necessary
 * @tparam T tensor type
 * @tparam O operation class
 * @return the tensor with the forward result
 */
template <typename TensorType>
std::shared_ptr<TensorType> Node<TensorType>::Evaluate(bool is_training)
{
  op_ptr_->SetTraining(is_training);

  if (cached_output_status_ != CachedOutputState::VALID_CACHE)
  {
    VecTensorType inputs = GatherInputs();

    if (cached_output_status_ == CachedOutputState::CHANGED_SIZE)
    {
      auto output_shape = op_ptr_->ComputeOutputShape(inputs);

      if (cached_output_.shape() !=
          output_shape)  // make shape compatible right before we do the forwarding
      {
        cached_output_.Reshape(output_shape);
      }
    }

    op_ptr_->Forward(inputs, cached_output_);
    cached_output_status_ = CachedOutputState::VALID_CACHE;

    if (math::state_division_by_zero<DataType>())
    {
      throw std::runtime_error("Division by zero encountered in Node::Evaluate");
    }
    if (math::state_infinity<DataType>())
    {
      throw std::runtime_error("Infinity encountered in Node::Evaluate");
    }
    if (math::state_nan<DataType>())
    {
      throw std::runtime_error("NaN encountered in Node::Evaluate");
    }

    assert(!math::state_overflow<DataType>());
  }

  return std::make_shared<TensorType>(cached_output_);
}

/**
 * Recursively backpropagates error_signal through this node to all input nodes
 * @tparam T the tensor type
 * @tparam O the operation class
 * @param error_signal the error signal to backpropagate
 * @return
 */
template <typename TensorType>
typename Node<TensorType>::NodeErrorMapType Node<TensorType>::BackPropagate(
    TensorType const &error_signal)
{
  NodeErrorMapType ret;

  // gather inputs and backprop for this node
  std::vector<TensorType> error_signals = op_ptr_->Backward(GatherInputs(), error_signal);
  assert(error_signals.size() == GatherInputs().size() || GatherInputs().empty());

  if (input_nodes_.empty())
  {
    // if this node has no inputs assign error signal to this node
    ret[this] = error_signals;
  }
  else
  {
    // otherwise backpropagate on the input nodes
    auto bp_it = error_signals.begin();
    for (auto &i : input_nodes_)
    {
      if (auto ptr = i.lock())
      {
        auto ret_err_sig = ptr->BackPropagate(*bp_it);
        ret.insert(ret_err_sig.begin(), ret_err_sig.end());
      }
      else
      {
        throw std::runtime_error("Unable to lock weak pointer.");
      }

      ++bp_it;
    }
  }

  if (math::state_division_by_zero<DataType>())
  {
    throw std::runtime_error("Division by zero encountered in Node::BackPropagate");
  }
  if (math::state_infinity<DataType>())
  {
    throw std::runtime_error("Infinity encountered in Node::BackPropagate");
  }
  if (math::state_nan<DataType>())
  {
    throw std::runtime_error("NaN encountered in Node::BackPropagate");
  }

  assert(!math::state_overflow<DataType>());
  return ret;
}
/**
 * Resets input and output node ptr containers. Useful for graph decompiling.
 * @tparam T
 */
template <typename TensorType>
void Node<TensorType>::ResetInputsAndOutputs()
{
  input_nodes_.clear();
  outputs_.clear();
}

/**
 * registers a node as an input to this node
 * @tparam T tensor type
 * @tparam O operation class
 * @param i pointer to the input node
 */
template <typename TensorType>
void Node<TensorType>::AddInput(NodeWeakPtrType const &i)
{
  input_nodes_.push_back(i);
}

/**
 * registers a node as an input to this node
 * @tparam T tensor type
 * @tparam O operation class
 * @param i pointer to the input node
 */
template <typename TensorType>
std::vector<std::string> Node<TensorType>::GetInputNames()
{
  std::vector<std::string> ret{};
  for (auto const &input_node : input_nodes_)
  {
    if (auto ptr = input_node.lock())
    {
      ret.emplace_back(ptr->name_);
    }
    else
    {
      throw std::runtime_error("Unable to lock weak pointer.");
    }
  }
  return ret;
}

/**
 * registers a node as an output to this node
 * @tparam T tensor type
 * @tparam O operation class
 * @param o pointer to the output node
 */
template <typename TensorType>
void Node<TensorType>::AddOutput(NodeWeakPtrType const &o)
{
  outputs_.push_back(o);
}

/**
 * gets all registered outputs of this node
 * @tparam T tensor type
 * @tparam O operation class
 * @return vector of pointers to output nodes
 */
template <typename TensorType>
std::vector<typename Node<TensorType>::NodeWeakPtrType> const &Node<TensorType>::GetOutputs() const
{
  return outputs_;
}

/**
 * Resets the cache status of this node depending on whether the input size has changed
 * @tparam T tensor type
 * @tparam O operation class
 * @param input_size_changed boolean indicating whether the input size changed
 */
template <typename TensorType>
void Node<TensorType>::ResetCache(bool input_size_changed)
{
  if (cached_output_status_ != CachedOutputState::CHANGED_SIZE)
  {
    cached_output_status_ =
        input_size_changed ? CachedOutputState::CHANGED_SIZE : CachedOutputState::CHANGED_CONTENT;
  }
}

/**
 * Sets the saveable params back to the node
 * @tparam TensorType
 * @tparam OperationType
 * @param nsp
 * @param op_ptr
 */
template <typename TensorType>
void Node<TensorType>::SetNodeSaveableParams(NodeSaveableParams<TensorType> const &nsp,
                                             std::shared_ptr<ops::Ops<TensorType>> op_ptr)
{
  name_                 = nsp.name;
  cached_output_status_ = CachedOutputState::CHANGED_SIZE;
  operation_type_       = nsp.operation_type;
  op_ptr_               = op_ptr;
}

}  // namespace ml
}  // namespace fetch
