/*
 * im2row.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */
#include <Avocado/reference_backend.h>

#include <Avocado/backend_descriptors.hpp>
#include "utils.hpp"
#include "activations.hpp"

namespace
{
	using namespace avocado::backend;

	struct int4
	{
			int x0, x1, x2, x3;
	};

	template<typename T>
	void kernel_im2row_1d(const reference::ConvolutionDescriptor &config, const reference::TensorDescriptor &rowDesc, T *rowMem,
			const reference::TensorDescriptor &srcDesc, const T *srcMem, const reference::TensorDescriptor &filterDesc)
	{
		const int batch_size = srcDesc.dimension(0);
		const int input_height = srcDesc.dimension(1);

		const int filter_height = filterDesc.dimension(1);
		const int input_filters = filterDesc.lastDim();

		const int padding_h = config.padding[0];
		const int stride_h = config.stride[0];
		const int dilation_h = config.dilation[0];

		const T padding_value = reference::getScalarValue<T>(config.padding_value.data());

		reference::TensorDescriptor output_shape = config.getOutputShape(srcDesc, filterDesc);

		int tile_idx = 0;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < output_shape.dimension(1); h++, tile_idx++)
			{
				int tmp_idx = 0;
				for (int i = 0; i < filter_height; i++)
				{
					int x;
					if (config.mode == AVOCADO_CONVOLUTION_MODE)
						x = padding_h + i * dilation_h + h * stride_h;
					else
						x = padding_h + (filter_height - 1 - i) * dilation_h + h * stride_h;
					if (x >= 0 and x < input_height)
					{
						for (int f = 0; f < input_filters; f++, tmp_idx++)
							rowMem[rowDesc.getIndex( { tile_idx, tmp_idx })] = srcMem[srcDesc.getIndex( { b, x, f })];
					}
					else
					{
						for (int f = 0; f < input_filters; f++, tmp_idx++)
							rowMem[rowDesc.getIndex( { tile_idx, tmp_idx })] = padding_value;
					}
				}
			}
	}

	template<typename T>
	void kernel_im2row_2d(const reference::ConvolutionDescriptor &config, const reference::TensorDescriptor &rowDesc, T *rowMem,
			const reference::TensorDescriptor &srcDesc, const T *srcMem, const reference::TensorDescriptor &filterDesc)
	{
		const int batch_size = srcDesc.dimension(0);
		const int input_height = srcDesc.dimension(1);
		const int input_width = srcDesc.dimension(2);

		const int filter_height = filterDesc.dimension(1);
		const int filter_width = filterDesc.dimension(2);
		const int input_filters = filterDesc.lastDim();

		const int padding_h = config.padding[0];
		const int padding_w = config.padding[1];

		const int stride_h = config.stride[0];
		const int stride_w = config.stride[1];

		const int dilation_h = config.dilation[0];
		const int dilation_w = config.dilation[1];

		const T padding_value = reference::getScalarValue<T>(config.padding_value.data());

		reference::TensorDescriptor output_shape = config.getOutputShape(srcDesc, filterDesc);

		int tile_idx = 0;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < output_shape.dimension(1); h++)
				for (int w = 0; w < output_shape.dimension(2); w++, tile_idx++)
				{
					int tmp_idx = 0;
					for (int i = 0; i < filter_height; i++)
						for (int j = 0; j < filter_width; j++)
						{
							int x, y;
							if (config.mode == AVOCADO_CONVOLUTION_MODE)
							{
								x = padding_h + i * dilation_h + h * stride_h;
								y = padding_w + j * dilation_w + w * stride_w;
							}
							else
							{
								x = padding_h + (filter_height - 1 - i) * dilation_h + h * stride_h;
								y = padding_w + (filter_width - 1 - j) * dilation_w + w * stride_w;
							}
							if (x >= 0 and x < input_height and y >= 0 and y < input_width)
							{
								for (int f = 0; f < input_filters; f++, tmp_idx++)
									rowMem[rowDesc.getIndex( { tile_idx, tmp_idx })] = srcMem[srcDesc.getIndex( { b, x, y, f })];
							}
							else
							{
								for (int f = 0; f < input_filters; f++, tmp_idx++)
									rowMem[rowDesc.getIndex( { tile_idx, tmp_idx })] = padding_value;
							}
						}
				}
	}

	template<typename T>
	avStatus_t launcher_im2row(const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc, const avTensorDescriptor_t srcDesc,
			const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t rowDesc, avMemoryDescriptor_t rowMem)
	{
		switch (reference::getTensor(filterDesc).nbDims())
		{
			case 3: // 1D convolution
				kernel_im2row_1d(reference::getConvolution(config), reference::getTensor(rowDesc), reference::getPointer<T>(rowMem),
						reference::getTensor(srcDesc), reference::getPointer<T>(srcMem), reference::getTensor(filterDesc));
				return AVOCADO_STATUS_SUCCESS;
			case 4: // 2D convolution
				kernel_im2row_2d(reference::getConvolution(config), reference::getTensor(rowDesc), reference::getPointer<T>(rowMem),
						reference::getTensor(srcDesc), reference::getPointer<T>(srcMem), reference::getTensor(filterDesc));
				return AVOCADO_STATUS_SUCCESS;
			case 5: // 3D convolution
				return AVOCADO_STATUS_NOT_SUPPORTED; // TODO
			default:
				return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refIm2Row(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t rowDesc,
				avMemoryDescriptor_t rowMem)
		{
			switch (reference::dataTypeSize(reference::getTensor(srcDesc).dtype()))
			{
				case 1:
					return launcher_im2row<int8_t>(config, filterDesc, srcDesc, srcMem, rowDesc, rowMem);
				case 2:
					return launcher_im2row<int16_t>(config, filterDesc, srcDesc, srcMem, rowDesc, rowMem);
				case 4:
					return launcher_im2row<int32_t>(config, filterDesc, srcDesc, srcMem, rowDesc, rowMem);
				case 8:
					return launcher_im2row<int64_t>(config, filterDesc, srcDesc, srcMem, rowDesc, rowMem);
				case 16:
					return launcher_im2row<int4>(config, filterDesc, srcDesc, srcMem, rowDesc, rowMem);
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}

	} /* namespace backend */
} /* namespace avocado */

