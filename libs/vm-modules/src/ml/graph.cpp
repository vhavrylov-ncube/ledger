//------------------------------------------------------------------------------
//
//   Copyright 2018-2019 Fetch.AI Limited
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

#include "ml/core/graph.hpp"

#include "core/byte_array/decoders.hpp"
#include "ml/layers/convolution_1d.hpp"
#include "ml/layers/fully_connected.hpp"
#include "ml/ops/activation.hpp"
#include "ml/ops/loss_functions/cross_entropy_loss.hpp"
#include "ml/ops/loss_functions/mean_square_error_loss.hpp"
#include "ml/saveparams/saveable_params.hpp"
#include "ml/utilities/graph_builder.hpp"
#include "vm/module.hpp"
#include "vm_modules/math/tensor.hpp"
#include "vm_modules/ml/graph.hpp"
#include "vm_modules/ml/state_dict.hpp"

using namespace fetch::vm;

namespace fetch {
namespace vm_modules {
namespace ml {

using SizeType       = fetch::math::SizeType;
using MathTensorType = fetch::math::Tensor<VMGraph::DataType>;
using VMTensorType   = fetch::vm_modules::math::VMTensor;
using VMPtrString    = Ptr<String>;

using namespace std::placeholders;

using ActivationAdder = std::function<void(std::string const &, std::string const &)>;
using LossAdder =
    std::function<void(std::string const &, std::string const &, std::string const &)>;

std::map<std::string, SupportedLayerType> const VMGraph::layer_types_{
    {"placeholder", SupportedLayerType::PLACEHOLDER},
    {"fully_connected", SupportedLayerType::FULLY_CONNECTED},
    {"conv1d", SupportedLayerType::CONV1D},
    {"relu", SupportedLayerType::ACTIVATION_RELU},
    {"softmax", SupportedLayerType::ACTIVATION_SOFTMAX},
    {"mse", SupportedLayerType::LOSS_MSE},
    {"crossentropy", SupportedLayerType::LOSS_CROSSENTROPY},
    {"dropout", SupportedLayerType::DROPOUT},
    {"transpose", SupportedLayerType::TRANSPOSE},
    {"exp", SupportedLayerType::EXP},
};

VMGraph::VMGraph(VM *vm, TypeId type_id)
  : Object(vm, type_id)
{}

Ptr<VMGraph> VMGraph::Constructor(VM *vm, TypeId type_id)
{
  return Ptr<VMGraph>{new VMGraph(vm, type_id)};
}

void VMGraph::SetInput(VMPtrString const &name, Ptr<VMTensorType> const &input)
{
  graph_.SetInput(name->string(), (*input).GetTensor());
}

Ptr<VMTensorType> VMGraph::Evaluate(VMPtrString const &name)
{
  MathTensorType    t   = graph_.Evaluate(name->string(), false);
  Ptr<VMTensorType> ret = this->vm_->CreateNewObject<math::VMTensor>(t.shape());
  (*ret).Copy(t);
  return ret;
}

void VMGraph::BackPropagate(VMPtrString const &name)
{
  graph_.BackPropagate(name->string());
}

void VMGraph::Step(DataType const &lr)
{
  auto grads = graph_.GetGradients();
  for (auto &grad : grads)
  {
    grad *= static_cast<DataType>(-lr);
  }
  graph_.ApplyGradients(grads);
}

void VMGraph::AddPlaceholder(const fetch::vm::Ptr<String> &name)
{
  graph_.AddNode<fetch::ml::ops::PlaceHolder<MathTensorType>>(name->string(), {});
}

void VMGraph::AddFullyConnected(VMPtrString const &name, VMPtrString const &input_name, int in,
                                int out)
{
  graph_.AddNode<fetch::ml::layers::FullyConnected<MathTensorType>>(
      name->string(), {input_name->string()}, std::size_t(in), std::size_t(out));
}

void VMGraph::AddConv1D(VMPtrString const &name, VMPtrString const &input_name, int filters,
                        int in_channels, int kernel_size, int stride_size)
{
  graph_.AddNode<fetch::ml::layers::Convolution1D<MathTensorType>>(
      name->string(), {input_name->string()}, static_cast<SizeType>(filters),
      static_cast<SizeType>(in_channels), static_cast<SizeType>(kernel_size),
      static_cast<SizeType>(stride_size));
}

void VMGraph::AddRelu(VMPtrString const &name, VMPtrString const &input_name)
{
  graph_.AddNode<fetch::ml::ops::Relu<MathTensorType>>(name->string(), {input_name->string()});
}

void VMGraph::AddSoftmax(VMPtrString const &name, VMPtrString const &input_name)
{
  graph_.AddNode<fetch::ml::ops::Softmax<fetch::math::Tensor<DataType>>>(name->string(),
                                                                         {input_name->string()});
}

void VMGraph::AddCrossEntropyLoss(VMPtrString const &name, VMPtrString const &input_name,
                                  VMPtrString const &label_name)
{
  graph_.AddNode<fetch::ml::ops::CrossEntropyLoss<fetch::math::Tensor<DataType>>>(
      name->string(), {input_name->string(), label_name->string()});
}

void VMGraph::AddMeanSquareErrorLoss(VMPtrString const &name, VMPtrString const &input_name,
                                     VMPtrString const &label_name)
{
  graph_.AddNode<fetch::ml::ops::MeanSquareErrorLoss<fetch::math::Tensor<DataType>>>(
      name->string(), {input_name->string(), label_name->string()});
}

void VMGraph::AddDropout(VMPtrString const &name, VMPtrString const &input_name,
                         DataType const &prob)
{
  graph_.AddNode<fetch::ml::ops::Dropout<MathTensorType>>(name->string(), {input_name->string()},
                                                          prob);
}

void VMGraph::AddTranspose(VMPtrString const &name, VMPtrString const &input_name)
{
  graph_.AddNode<fetch::ml::ops::Transpose<MathTensorType>>(name->string(), {input_name->string()});
}

void VMGraph::AddExp(VMPtrString const &name, VMPtrString const &input_name)
{
  graph_.AddNode<fetch::ml::ops::Exp<MathTensorType>>(name->string(), {input_name->string()});
}

void VMGraph::LoadStateDict(Ptr<VMStateDict> const &sd)
{
  graph_.LoadStateDict(sd->state_dict_);
}

Ptr<VMStateDict> VMGraph::StateDict()
{
  Ptr<VMStateDict> ret = this->vm_->CreateNewObject<VMStateDict>(graph_.StateDict());
  return ret;
}

void VMGraph::Bind(Module &module)
{
  using VMPtrStringCRef = VMPtrString const &;
  module.CreateClassType<VMGraph>("Graph")
      .CreateConstructor(&VMGraph::Constructor)
      .CreateSerializeDefaultConstructor([](VM *vm, TypeId type_id) -> Ptr<VMGraph> {
        return Ptr<VMGraph>{new VMGraph(vm, type_id)};
      })
      .CreateMemberFunction("setInput", &VMGraph::SetInput)
      .CreateMemberFunction("evaluate", &VMGraph::Evaluate)
      .CreateMemberFunction("backPropagate", &VMGraph::BackPropagate)
      .CreateMemberFunction("step", &VMGraph::Step)
      .CreateMemberFunction("add", &VMGraph::AddLayer<>)                 // AddPlaceholder
      .CreateMemberFunction("add", &VMGraph::AddLayer<VMPtrStringCRef>)  // AddSoftmax, AddRelu
      .CreateMemberFunction(
          "add", &VMGraph::AddLayer<VMPtrStringCRef, VMPtrStringCRef>)  // AddMeanSquareErrorLoss,
                                                                        // AddCrossEntropyLoss
      .CreateMemberFunction("addPlaceholder", &VMGraph::AddPlaceholder)
      .CreateMemberFunction("addFullyConnected", &VMGraph::AddFullyConnected)
      .CreateMemberFunction("addConv1D", &VMGraph::AddConv1D)
      .CreateMemberFunction("addRelu", &VMGraph::AddRelu)
      .CreateMemberFunction("addSoftmax", &VMGraph::AddSoftmax)
      .CreateMemberFunction("addDropout", &VMGraph::AddDropout)
      .CreateMemberFunction("addCrossEntropyLoss", &VMGraph::AddCrossEntropyLoss)
      .CreateMemberFunction("addMeanSquareErrorLoss", &VMGraph::AddMeanSquareErrorLoss)
      .CreateMemberFunction("addTranspose", &VMGraph::AddTranspose)
      .CreateMemberFunction("addExp", &VMGraph::AddExp)
      .CreateMemberFunction("loadStateDict", &VMGraph::LoadStateDict)
      .CreateMemberFunction("stateDict", &VMGraph::StateDict)
      .CreateMemberFunction("serializeToString", &VMGraph::SerializeToString)
      .CreateMemberFunction("deserializeFromString", &VMGraph::DeserializeFromString);
}

VMGraph::GraphType &VMGraph::GetGraph()
{
  return graph_;
}

bool VMGraph::SerializeTo(serializers::MsgPackSerializer &buffer)
{
  buffer << graph_.GetGraphSaveableParams();
  return true;
}

bool VMGraph::DeserializeFrom(serializers::MsgPackSerializer &buffer)
{
  fetch::ml::GraphSaveableParams<fetch::math::Tensor<fetch::vm_modules::math::DataType>> gsp;
  buffer >> gsp;

  auto vm_graph  = std::make_shared<fetch::vm_modules::ml::VMGraph>(this->vm_, this->type_id_);
  auto graph_ptr = std::make_shared<fetch::ml::Graph<MathTensorType>>(vm_graph->GetGraph());
  fetch::ml::utilities::BuildGraph<MathTensorType>(gsp, graph_ptr);

  vm_graph->GetGraph() = *graph_ptr;
  *this                = *vm_graph;

  return true;
}

fetch::vm::Ptr<fetch::vm::String> VMGraph::SerializeToString()
{
  serializers::MsgPackSerializer b;
  SerializeTo(b);
  auto byte_array_data = b.data().ToBase64();
  return Ptr<String>{new fetch::vm::String(vm_, static_cast<std::string>(byte_array_data))};
}

fetch::vm::Ptr<VMGraph> VMGraph::DeserializeFromString(
    fetch::vm::Ptr<fetch::vm::String> const &graph_string)
{
  byte_array::ConstByteArray b(graph_string->string());
  b = byte_array::FromBase64(b);
  MsgPackSerializer buffer(b);
  DeserializeFrom(buffer);

  auto vm_graph        = fetch::vm::Ptr<VMGraph>(new VMGraph(vm_, type_id_));
  vm_graph->GetGraph() = graph_;

  return vm_graph;
}

void VMGraph::AssertLayerTypeMatches(SupportedLayerType                layer,
                                     std::vector<SupportedLayerType> &&valids) const
{
  static const std::map<SupportedLayerType, std::string> LAYER_NAMES_{
      {SupportedLayerType::PLACEHOLDER, "placeholder"},
      {SupportedLayerType::FULLY_CONNECTED, "fully_connected"},
      {SupportedLayerType::CONV1D, "conv1d"},
      {SupportedLayerType::ACTIVATION_RELU, "relu"},
      {SupportedLayerType::ACTIVATION_SOFTMAX, "softmax"},
      {SupportedLayerType::LOSS_MSE, "mse"},
      {SupportedLayerType::LOSS_CROSSENTROPY, "crossentropy"},
      {SupportedLayerType::DROPOUT, "dropout"},
      {SupportedLayerType::TRANSPOSE, "transpose"},
      {SupportedLayerType::EXP, "exp"},
  };
  if (std::find(valids.begin(), valids.end(), layer) == valids.end())
  {
    throw std::runtime_error("Invalid params specified for \"" + LAYER_NAMES_.at(layer) +
                             "\" layer.");
  }
}

/**
 * Converts between user specified string and output type (e.g. activation, layer etc.)
 * invokes VM runtime error if parsing failed.
 * @param name user specified string to convert
 * @param dict dictionary of existing entities
 * @param errmsg preferred display name of expected type, that was not parsed
 */
template <typename T>
inline T VMGraph::ParseName(std::string const &name, std::map<std::string, T> const &dict,
                            std::string const &errmsg) const
{
  if (dict.find(name) == dict.end())
  {
    throw std::runtime_error("Unknown " + errmsg + " name : " + name);
  }
  return dict.at(name);
}

void VMGraph::AddLayerSpecificImpl(SupportedLayerType type, fetch::vm::Ptr<String> const &name)
{
  AssertLayerTypeMatches(type, {SupportedLayerType::PLACEHOLDER});
  graph_.AddNode<fetch::ml::ops::PlaceHolder<MathTensorType>>(name->string(), {});
}

void VMGraph::AddLayerSpecificImpl(SupportedLayerType type, fetch::vm::Ptr<String> const &name,
                                   fetch::vm::Ptr<String> const &input_name)
{
  AssertLayerTypeMatches(
      type, {SupportedLayerType::ACTIVATION_RELU, SupportedLayerType::ACTIVATION_SOFTMAX});

  static std::map<SupportedLayerType, ActivationAdder> const activation_adders_{
      {SupportedLayerType::ACTIVATION_RELU,
       std::bind(&VMGraph::AddActivation<fetch::ml::ops::Relu<fetch::math::Tensor<DataType>>>, this,
                 _1, _2)},
      {SupportedLayerType::ACTIVATION_SOFTMAX,
       std::bind(&VMGraph::AddActivation<fetch::ml::ops::Softmax<fetch::math::Tensor<DataType>>>,
                 this, _1, _2)},
  };

  activation_adders_.at(type)(name->string(), input_name->string());
}

void VMGraph::AddLayerSpecificImpl(SupportedLayerType type, fetch::vm::Ptr<String> const &name,
                                   fetch::vm::Ptr<String> const &input_name,
                                   fetch::vm::Ptr<String> const &labels)
{
  AssertLayerTypeMatches(type,
                         {SupportedLayerType::LOSS_MSE, SupportedLayerType::LOSS_CROSSENTROPY});

  static std::map<SupportedLayerType, LossAdder> const loss_adders_{
      {SupportedLayerType::LOSS_MSE,
       std::bind(&VMGraph::AddLoss<fetch::ml::ops::Relu<fetch::math::Tensor<DataType>>>, this, _1,
                 _2, _3)},
      {SupportedLayerType::LOSS_CROSSENTROPY,
       std::bind(&VMGraph::AddLoss<fetch::ml::ops::Softmax<fetch::math::Tensor<DataType>>>, this,
                 _1, _2, _3)},
  };

  loss_adders_.at(type)(name->string(), input_name->string(), labels->string());
}

template <typename... LayerArgs>
void VMGraph::AddLayer(const fetch::vm::Ptr<String> &type, const fetch::vm::Ptr<String> &name,
                       LayerArgs... args)
{
  try
  {
    SupportedLayerType const layer_type = ParseName(type->string(), layer_types_, "layer type");
    AddLayerSpecificImpl(layer_type, name, args...);
  }
  catch (std::exception &e)
  {
    vm_->RuntimeError("Impossible to add layer : " + std::string(e.what()));
    return;
  }
}

template <typename ActivationType>
void VMGraph::AddActivation(std::string const &name, std::string const &input_name)
{
  graph_.AddNode<ActivationType>(name, {input_name});
}

template <typename LossType>
void VMGraph::AddLoss(std::string const &name, std::string const &input_name,
                      std::string const &label_name)
{
  graph_.AddNode<LossType>(name, {input_name, label_name});
}

}  // namespace ml
}  // namespace vm_modules
}  // namespace fetch
