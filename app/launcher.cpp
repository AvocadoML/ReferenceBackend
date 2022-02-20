/*
 * launcher.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

#include "../src/fp16.hpp"
#include "../src/utils.hpp"
#include "../src/activations.hpp"

#include <initializer_list>
#include <stddef.h>
#include <iostream>

using namespace avocado::backend;

#define CHECK_STATUS(status) if (status != AVOCADO_STATUS_SUCCESS) throw std::runtime_error(__FUNCTION__);

class TensorWrapper
{
	private:
		avTensorDescriptor_t desc;
		avMemoryDescriptor_t mem;
	public:
		TensorWrapper(std::initializer_list<int> dimensions, avDataType_t dtype)
		{
			avStatus_t status = refCreateTensorDescriptor(&(this->desc));
			CHECK_STATUS(status);
			status = refSetTensorDescriptor(desc, dtype, dimensions.size(), dimensions.begin());
			CHECK_STATUS(status);
			status = refCreateMemoryDescriptor(&(this->mem), reference::getTensor(desc).sizeInBytes());
			CHECK_STATUS(status);
			status = refSetMemory(0, mem, 0, reference::getTensor(desc).sizeInBytes(), nullptr, 0);
			CHECK_STATUS(status);
		}
		~TensorWrapper()
		{
			refDestroyTensorDescriptor(desc);
			refDestroyMemoryDescriptor(mem);
		}
		template<typename T>
		void fill(T value)
		{
			refSetMemory(0, mem, 0, reference::getTensor(desc).sizeInBytes(), &value, sizeof(value));
		}
		template<typename T>
		T& at(std::initializer_list<int> idx)
		{
			return *(reinterpret_cast<T*>(refGetMemoryPointer(desc)) + reference::getTensor(desc).getIndex(idx));
		}
		template<typename T>
		T at(std::initializer_list<int> idx) const
		{
			return *(reinterpret_cast<const T*>(refGetMemoryPointer(desc)) + reference::getTensor(desc).getIndex(idx));
		}
		avTensorDescriptor_t getDesc() const noexcept
		{
			return desc;
		}
		avMemoryDescriptor_t getMem() const noexcept
		{
			return mem;
		}
};

class ContextWrapper
{
	private:
		avContextDescriptor_t desc;
	public:
		ContextWrapper()
		{
			refCreateContextDescriptor(&(this->desc));
		}
		~ContextWrapper()
		{
			refDestroyContextDescriptor(desc);
		}
		avContextDescriptor_t getDesc() const noexcept
		{
			return desc;
		}
};

template<typename T>
T unary_op(avUnaryOp_t operation, T x) noexcept
{
	switch (operation)
	{
		case AVOCADO_UNARY_OP_ABS:
			return avocado::backend::abs(x);
		case AVOCADO_UNARY_OP_CEIL:
			return avocado::backend::ceil(x);
		case AVOCADO_UNARY_OP_COS:
			return avocado::backend::cos(x);
		case AVOCADO_UNARY_OP_EXP:
			return avocado::backend::exp(x);
		case AVOCADO_UNARY_OP_FLOOR:
			return avocado::backend::floor(x);
		case AVOCADO_UNARY_OP_LN:
			return avocado::backend::log(x);
		case AVOCADO_UNARY_OP_NEG:
			return -x;
		case AVOCADO_UNARY_OP_RCP:
			return one<T>() / x;
		case AVOCADO_UNARY_OP_RSQRT:
			return one<T>() / avocado::backend::sqrt(x);
		case AVOCADO_UNARY_OP_SIN:
			return avocado::backend::sin(x);
		case AVOCADO_UNARY_OP_SQUARE:
			return avocado::backend::square(x);
		case AVOCADO_UNARY_OP_SQRT:
			return avocado::backend::sqrt(x);
		case AVOCADO_UNARY_OP_TAN:
			return avocado::backend::tan(x);
		case AVOCADO_UNARY_OP_LOGICAL_NOT:
			return LogicalNot<T>::value(x);
	}
	return zero<T>();
}
template<typename T, typename U>
void kernel_unary_op(T *dst, const T *src, U alpha, U beta, avSize_t elements, avUnaryOp_t operation) noexcept
{
	if (beta == zero<U>())
		clear(dst, elements);

	for (avSize_t i = 0; i < elements; i++)
	{
		T value = static_cast<T>(alpha * static_cast<U>(src[i]));
		T result = unary_op(operation, value);
		U tmp = static_cast<U>(result);
		dst[i] = tmp;
//		dst[i] = static_cast<U>(result);//+ beta * static_cast<U>(dst[i]);
	}
}
template<typename T>
void kernel_unary_logical_op(T *dst, const T *src, avSize_t elements, avUnaryOp_t operation) noexcept
{
	clear(dst, elements);
	for (avSize_t i = 0; i < elements; i++)
		dst[i] = ~(src[i]);
}

void unaryOp(avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *beta,
		const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
{
	const avSize_t elements = reference::getTensor(aDesc).volume();
	kernel_unary_op(reference::getPointer<bfloat16>(cMem), reference::getPointer<bfloat16>(aMem), reference::getAlphaValue(alpha),
			reference::getBetaValue(beta), elements, operation);
}

int main(int argc, char *argv[])
{
	avConvolutionDescriptor_t config;
	refCreateConvolutionDescriptor(&config);
	std::array<int, 3> padding = { -1, -1, 0 };
	std::array<int, 3> strides = { 1, 1, 0 };
	std::array<int, 3> dilation = { 1, 1, 0 };
	refSetConvolutionDescriptor(config, AVOCADO_CONVOLUTION_MODE, 2, padding.data(), strides.data(), dilation.data(), 1, nullptr);

	TensorWrapper weight( { 13, 3, 3, 1 }, AVOCADO_DTYPE_FLOAT32);
	TensorWrapper input( { 1, 2, 2, 1 }, AVOCADO_DTYPE_FLOAT32);
	TensorWrapper matrices( { 16, 1, 1 }, AVOCADO_DTYPE_FLOAT32);
	refWinogradInputTransform(0, config, 2, weight.getDesc(), input.getDesc(), input.getMem(), matrices.getDesc(), matrices.getMem());

//	TensorWrapper input( { 123 }, AVOCADO_DTYPE_BFLOAT16);
//	TensorWrapper output( { 123 }, AVOCADO_DTYPE_BFLOAT16);
//	input.fill(bfloat16(1.23f));
//	unaryOp(AVOCADO_UNARY_OP_NEG, nullptr, input.getDesc(), input.getMem(), nullptr, output.getDesc(), output.getMem());

//	TensorWrapper tensor( { 11, 12, 12, 9 }, reference::typeOf<float>());
//	TensorWrapper matrices( { 36, 9 * 11, 9 }, reference::typeOf<float>());
//	matrices.fill(1.0f);
//
//	ContextWrapper cw;
//	ConvolutionDescriptor config;
//	config.filter = createShapeDescriptor( { 3, 3 });
//	config.padding = createShapeDescriptor( { -1, -1 });
//
//	refWinogradInputTransform(nullptr, &config, 4, tensor, matrices);
//
////	for (int i = 0; i < 20; i++)
////		for (int j = 0; j < 9; j++)
////			tw.at<float>( { i, j }) = i * 10 + j;
//
//	for (int k = 0; k < 9; k++)
//	{
//		for (int i = 0; i < 6; i++)
//		{
//			for (int j = 0; j < 6; j++)
//				std::cout << matrices.at<float>( { i * 6 + j, k, 0 }) << ' ';
//			std::cout << '\n';
//		}
//		std::cout << '\n';
//	}

	std::cout << "END" << std::endl;
	return 0;
}

