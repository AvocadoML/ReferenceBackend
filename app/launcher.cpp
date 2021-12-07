/*
 * launcher.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "../src/descriptors.hpp"

#include <initializer_list>
#include <stddef.h>
#include <iostream>

using namespace avocado::backend;

class TensorWrapper
{
	private:
		avTensorDescriptor_t desc;
		avMemoryDescriptor_t mem;
	public:
		TensorWrapper(std::initializer_list<int> dimensions, avDataType_t dtype)
		{
			refCreateTensorDescriptor(&(this->desc));
			refSetTensorDescriptor(desc, dtype, dimensions.size(), dimensions.begin());
			refCreateMemoryDescriptor(&(this->mem), getTensor(desc).sizeInBytes());
			refSetMemory(0, mem, getTensor(desc).sizeInBytes(), nullptr, 0);
		}
		~TensorWrapper()
		{
			refDestroyTensorDescriptor(desc);
			refDestroyMemoryDescriptor(mem);
		}
		template<typename T>
		void fill(T value)
		{
			refSetMemory(0, desc, getTensor(desc).sizeInBytes(), &value, sizeof(value));
		}
		template<typename T>
		T& at(std::initializer_list<int> idx)
		{
			return *(reinterpret_cast<T*>(refGetMemoryPointer(desc)) + getTensor(desc).getIndex(idx));
		}
		template<typename T>
		T at(std::initializer_list<int> idx) const
		{
			return *(reinterpret_cast<const T*>(refGetMemoryPointer(desc)) + getTensor(desc).getIndex(idx));
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

int main(int argc, char *argv[])
{
	TensorWrapper tensor( { 11, 12, 12, 9 }, typeOf<float>());
	TensorWrapper matrices( { 36, 9 * 11, 9 }, typeOf<float>());
	matrices.fill(1.0f);

	ContextWrapper cw;
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

