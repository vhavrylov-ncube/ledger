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

#include "vm_modules/math/template_tensor.hpp"

#include "math/tensor.hpp"
#include "vm/array.hpp"
#include "vm/module.hpp"
#include "vm/object.hpp"
#include "vm_modules/math/type.hpp"

#include <cstdint>
#include <vector>

using namespace fetch::vm;

namespace fetch {
namespace vm_modules {
namespace math {

using ArrayType  = fetch::math::Tensor<TemplateTensor::DataType>;
using SizeType   = ArrayType::SizeType;
using SizeVector = ArrayType::SizeVector;

ITemplateTensor::ITemplateTensor(VM *vm, TypeId type_id)
  : Object(vm, type_id)
{}

Ptr<ITemplateTensor> ITemplateTensor::Constructor(fetch::vm::VM *vm, fetch::vm::TypeId type_id,
                                                  std::vector<uint64_t> const &shape)
{
  // TypeInfo const &type_info       = vm->GetTypeInfo(type_id);
  // TypeId const    element_type_id = type_info.template_parameter_type_ids[0];
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return Ptr<ITemplateTensor>{new TemplateTensor(vm, type_id, shape)};
}

void ITemplateTensor::Bind(Module &module)
{
  module.CreateTemplateType<ITemplateTensor, Any>("Matrix")
      .CreateConstructor(&ITemplateTensor::Constructor)
      //.EnableIndexOperator(&ITemplateTensor::GetIndexedValue, &ITemplateTensor::SetIndexedValue)
      .CreateInstantiationType<TemplateTensor>();
}

TemplateTensor::TemplateTensor(VM *vm, TypeId type_id, std::vector<uint64_t> const &shape)
  : ITemplateTensor(vm, type_id)
  , tensor_(shape)
{}

TemplateTensor::TemplateTensor(VM *vm, TypeId type_id, ArrayType tensor)
  : ITemplateTensor(vm, type_id)
  , tensor_(std::move(tensor))
{}

TemplateTensor::TemplateTensor(VM *vm, TypeId type_id)
  : ITemplateTensor(vm, type_id)
{}

Ptr<TemplateTensor> TemplateTensor::Constructor(VM *vm, TypeId type_id,
                                                Ptr<Array<SizeType>> const &shape)
{
  return Ptr<TemplateTensor>{new TemplateTensor(vm, type_id, shape->elements)};
}

void TemplateTensor::Bind(Module &module)
{
  using Index = fetch::math::SizeType;
  module.CreateClassType<TemplateTensor>("Tensor")
      .CreateConstructor(&TemplateTensor::Constructor)
      .CreateSerializeDefaultConstructor([](VM *vm, TypeId type_id) -> Ptr<TemplateTensor> {
        return Ptr<TemplateTensor>{new TemplateTensor(vm, type_id)};
      })
      .CreateMemberFunction("at", &TemplateTensor::At<Index>)
      .CreateMemberFunction("at", &TemplateTensor::At<Index, Index>)
      .CreateMemberFunction("at", &TemplateTensor::At<Index, Index, Index>)
      .CreateMemberFunction("at", &TemplateTensor::At<Index, Index, Index, Index>)
      .CreateMemberFunction("at", &TemplateTensor::At<Index, Index, Index, Index, Index>)
      .CreateMemberFunction("at", &TemplateTensor::At<Index, Index, Index, Index, Index, Index>)
      .CreateMemberFunction("setAt", &TemplateTensor::SetAt<Index, DataType>)
      .CreateMemberFunction("setAt", &TemplateTensor::SetAt<Index, Index, DataType>)
      .CreateMemberFunction("setAt", &TemplateTensor::SetAt<Index, Index, Index, DataType>)
      .CreateMemberFunction("setAt", &TemplateTensor::SetAt<Index, Index, Index, Index, DataType>)
      .CreateMemberFunction("setAt",
                            &TemplateTensor::SetAt<Index, Index, Index, Index, Index, DataType>)
      .CreateMemberFunction(
          "setAt", &TemplateTensor::SetAt<Index, Index, Index, Index, Index, Index, DataType>)
      .CreateMemberFunction("fill", &TemplateTensor::Fill)
      .CreateMemberFunction("fillRandom", &TemplateTensor::FillRandom)
      .CreateMemberFunction("reshape", &TemplateTensor::Reshape)
      .CreateMemberFunction("squeeze", &TemplateTensor::Squeeze)
      .CreateMemberFunction("size", &TemplateTensor::size)
      .CreateMemberFunction("transpose", &TemplateTensor::Transpose)
      .CreateMemberFunction("unsqueeze", &TemplateTensor::Unsqueeze)
      .CreateMemberFunction("fromString", &TemplateTensor::FromString)
      .CreateMemberFunction("toString", &TemplateTensor::ToString);

  // Add support for Array of Tensors
  module.GetClassInterface<IArray>().CreateInstantiationType<Array<Ptr<TemplateTensor>>>();
}

SizeVector TemplateTensor::shape() const
{
  return tensor_.shape();
}

SizeType TemplateTensor::size() const
{
  return tensor_.size();
}

////////////////////////////////////
/// ACCESSING AND SETTING VALUES ///
////////////////////////////////////

template <typename... Indices>
TemplateTensor::DataType TemplateTensor::At(Indices... indices) const
{
  return tensor_.At(indices...);
}

template <typename... Args>
void TemplateTensor::SetAt(Args... args)
{
  tensor_.Set(args...);
}

void TemplateTensor::Copy(ArrayType const &other)
{
  tensor_.Copy(other);
}

void TemplateTensor::Fill(DataType const &value)
{
  tensor_.Fill(value);
}

void TemplateTensor::FillRandom()
{
  tensor_.FillUniformRandom();
}

Ptr<TemplateTensor> TemplateTensor::Squeeze()
{
  auto squeezed_tensor = tensor_.Copy();
  squeezed_tensor.Squeeze();
  return fetch::vm::Ptr<TemplateTensor>(new TemplateTensor(vm_, type_id_, squeezed_tensor));
}

Ptr<TemplateTensor> TemplateTensor::Unsqueeze()
{
  auto unsqueezed_tensor = tensor_.Copy();
  unsqueezed_tensor.Unsqueeze();
  return fetch::vm::Ptr<TemplateTensor>(new TemplateTensor(vm_, type_id_, unsqueezed_tensor));
}

bool TemplateTensor::Reshape(Ptr<Array<SizeType>> const &new_shape)
{
  return tensor_.Reshape(new_shape->elements);
}

void TemplateTensor::Transpose()
{
  tensor_.Transpose();
}

//////////////////////////////
/// PRINTING AND EXPORTING ///
//////////////////////////////

void TemplateTensor::FromString(fetch::vm::Ptr<fetch::vm::String> const &string)
{
  tensor_.Assign(fetch::math::Tensor<DataType>::FromString(string->string()));
}

Ptr<String> TemplateTensor::ToString() const
{
  return Ptr<String>{new String(vm_, tensor_.ToString())};
}

ArrayType &TemplateTensor::GetTensor()
{
  return tensor_;
}

ArrayType const &TemplateTensor::GetConstTensor()
{
  return tensor_;
}

bool TemplateTensor::SerializeTo(serializers::MsgPackSerializer &buffer)
{
  buffer << tensor_;
  return true;
}

bool TemplateTensor::DeserializeFrom(serializers::MsgPackSerializer &buffer)
{
  buffer >> tensor_;
  return true;
}

}  // namespace math
}  // namespace vm_modules
}  // namespace fetch
