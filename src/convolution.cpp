/*
 * convolution.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "utils.hpp"
#include "descriptors.hpp"
#include "activations.hpp"

namespace
{
	using namespace avocado::backend;

	struct int2
	{
			int x0, x1;
	};

	struct int4
	{
			int x0, x1, x2, x3;
	};

	std::array<int, 3> getConvolutionOutputShape(const ConvolutionDescriptor &config, const TensorDescriptor &inputDesc,
			const TensorDescriptor &filterDesc)
	{
		std::array<int, 3> result;
		for (int i = 0; i < inputDesc.nbDims() - 2; i++)
			result[i] = 1
					+ (inputDesc.dimension(1 + i) - 2 * config.padding[i] - (((filterDesc.dimension(1 + i) - 1) * config.dilation[i]) + 1))
							/ config.stride[i];
		return result;
	}

	template<typename T>
	void kernel_im2col_2d(const ConvolutionDescriptor &config, const TensorDescriptor &colDesc, T *colMem, const TensorDescriptor &srcDesc,
			const T *srcMem, const TensorDescriptor &filterDesc)
	{
		const int batch_size = srcDesc.dimension(0);
		const int input_height = srcDesc.dimension(1);
		const int input_width = srcDesc.dimension(2);

		const int filter_height = filterDesc.dimension(1);
		const int filter_width = filterDesc.dimension(2);
		const int input_filters = filterDesc.dimension(3);

		const int padding_h = (config.mode == AVOCADO_CONVOLUTION_MODE) ? config.padding[0] : 1 - filter_height - config.padding[0];
		const int padding_w = (config.mode == AVOCADO_CONVOLUTION_MODE) ? config.padding[1] : 1 - filter_width - config.padding[1];

		const int stride_h = config.stride[0];
		const int stride_w = config.stride[1];

		const int dilation_h = config.dilation[0];
		const int dilation_w = config.dilation[1];

		const T padding_value = getScalarValue<T>(config.padding_value);

		std::array<int, 3> output_shape = getConvolutionOutputShape(config, srcDesc, filterDesc);

		int tile_idx = 0;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < output_shape[0]; h++)
				for (int w = 0; w < output_shape[1]; w++, tile_idx++)
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
									colMem[colDesc.getIndex( { tile_idx, tmp_idx })] = srcMem[srcDesc.getIndex( { b, x, y, f })];
							}
							else
							{
								for (int f = 0; f < input_filters; f++, tmp_idx++)
									colMem[colDesc.getIndex( { tile_idx, tmp_idx })] = padding_value;
							}
						}
				}
	}

	template<typename DataType, typename ComputeType = DataType, typename ScalingType = DataType, typename BiasType = DataType>
	void kernel_convolution_2d(const ConvolutionDescriptor &config, ScalingType alpha, const TensorDescriptor &xDesc, const DataType *xMem,
			const TensorDescriptor &wDesc, const DataType *wMem, ScalingType beta, const TensorDescriptor &yDesc, DataType *yMem,
			avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR, ScalingType alpha2 = zero<ScalingType>(), const BiasType *bias = nullptr,
			const DataType *zMem = nullptr)
	{
		const int batch_size = xDesc.dimension(0);

		const int output_filters = wDesc.dimension(0);
		const int filter_height = wDesc.dimension(1);
		const int filter_width = wDesc.dimension(2);
		const int input_filters = wDesc.dimension(3);

		const int padding_h = config.padding[0];
		const int padding_w = config.padding[1];

		const int stride_h = config.stride[0];
		const int stride_w = config.stride[1];

		const int dilation_h = config.dilation[0];
		const int dilation_w = config.dilation[1];

		const DataType padding_value = getScalarValue<DataType>(config.padding_value);

		if (beta == zero<ScalingType>())
			clear(yMem, yDesc.volume());

		for (int b = 0; b < batch_size; b++) // batch size
			for (int g = 0; g < config.groups; g++)
			{
				const int output_filters_group[2] = { g * output_filters / config.groups, (g + 1) * output_filters / config.groups };
				const int input_filters_group[2] = { g * input_filters / config.groups, (g + 1) * input_filters / config.groups };

				for (int out = output_filters_group[0]; out < output_filters_group[1]; out++) // output filters
					for (int out_h = 0; out_h < yDesc.dimension(1); out_h++) // output height
						for (int out_w = 0; out_w < yDesc.dimension(2); out_w++) // output width
						{
							ComputeType tmp = zero<ComputeType>();
							for (int i = 0; i < wDesc.dimension(1); i++) // kernel height
								for (int j = 0; j < wDesc.dimension(2); j++) // kernel width
								{
									int x, y;
									if (config.mode == AVOCADO_CONVOLUTION_MODE)
									{
										x = padding_h + i * dilation_h + out_h * stride_h;
										y = padding_w + j * dilation_w + out_w * stride_w;
									}
									else // AVOCADO_CROSS_CORRELATION_MODE
									{
										x = padding_h + (filter_height - 1 - i) * dilation_h + out_h * stride_h;
										y = padding_w + (filter_width - 1 - j) * dilation_w + out_w * stride_w;
									}
									if (x >= 0 and x < xDesc.dimension(1) and y >= 0 and y < xDesc.dimension(2))
									{
										for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
											tmp += static_cast<ComputeType>(wMem[wDesc.getIndex( { out, i, j, in })])
													* static_cast<ComputeType>(xMem[xDesc.getIndex( { b, x, y, in })]);
									}
									else
									{
										for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
											tmp += static_cast<ComputeType>(wMem[wDesc.getIndex( { out, i, j, in })])
													* static_cast<ComputeType>(padding_value);
									}
								}
							ScalingType tmp2 = alpha * static_cast<ScalingType>(tmp) + beta * static_cast<ScalingType>(yMem[yDesc.getIndex( { b,
									out_h, out_w, out })]);
							if (bias != nullptr)
								tmp2 += static_cast<ScalingType>(bias[out]);
							if (zMem != nullptr)
								tmp2 += alpha2 * static_cast<ScalingType>(zMem[yDesc.getIndex( { b, out_h, out_w, out })]);
							tmp2 = activation_forward(activation, tmp2);
							yMem[yDesc.getIndex( { b, out_h, out_w, out })] = Store<DataType, ScalingType>::store(tmp2);
						}
			}
	}

	template<typename T>
	void kernel_convolution_2d_update(const ConvolutionDescriptor &config, T alpha, const TensorDescriptor &xDesc, const T *xMem,
			const TensorDescriptor &dwDesc, T *dwMem, T beta, const TensorDescriptor &dyDesc, const T *dyMem)
	{
		const int batch_size = xDesc.dimension(0);

		const int output_filters = dwDesc.dimension(0);
		const int filter_height = dwDesc.dimension(1);
		const int filter_width = dwDesc.dimension(2);
		const int input_filters = dwDesc.dimension(3);

		const int padding_h = config.padding[0];
		const int padding_w = config.padding[1];

		const int stride_h = config.stride[0];
		const int stride_w = config.stride[1];

		const int dilation_h = config.dilation[0];
		const int dilation_w = config.dilation[1];

		const T padding_value = getScalarValue<T>(config.padding_value);

		if (beta == zero<T>())
			clear(dwMem, dwDesc.volume());
		else
		{
			for (avSize_t i = 0; i < dwDesc.volume(); i++)
				dwMem[i] *= beta;
		}

		for (int b = 0; b < batch_size; b++) // batch size
			for (int g = 0; g < config.groups; g++)
			{
				const int output_filters_group[2] = { g * output_filters / config.groups, (g + 1) * output_filters / config.groups };
				const int input_filters_group[2] = { g * input_filters / config.groups, (g + 1) * input_filters / config.groups };

				for (int out = output_filters_group[0]; out < output_filters_group[1]; out++) // output filters
					for (int out_h = 0; out_h < dyDesc.dimension(1); out_h++) // output height
						for (int out_w = 0; out_w < dyDesc.dimension(2); out_w++) // output width
						{
							for (int i = 0; i < dwDesc.dimension(1); i++) // kernel height
								for (int j = 0; j < dwDesc.dimension(2); j++) // kernel width
								{
									int x, y;
									if (config.mode == AVOCADO_CONVOLUTION_MODE)
									{
										x = padding_h + i * dilation_h + out_h * stride_h;
										y = padding_w + j * dilation_w + out_w * stride_w;
									}
									else // AVOCADO_CROSS_CORRELATION_MODE
									{
										x = padding_h + (filter_height - 1 - i) * dilation_h + out_h * stride_h;
										y = padding_w + (filter_width - 1 - j) * dilation_w + out_w * stride_w;
									}
									if (x >= 0 and x < xDesc.dimension(1) and y >= 0 and y < xDesc.dimension(2))
									{
										for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
											dwMem[dwDesc.getIndex( { out, i, j, in })] += alpha * dyMem[dyDesc.getIndex( { out, out_h, out_w, in })]
													* xMem[xDesc.getIndex( { b, x, y, in })];
									}
									else
									{
										for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
											dwMem[dwDesc.getIndex( { out, i, j, in })] += alpha * dyMem[dyDesc.getIndex( { out, out_h, out_w, in })]
													* padding_value;
									}
								}
						}
			}
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refIm2Col(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t colDesc,
				avMemoryDescriptor_t colMem)
		{
			switch (dataTypeSize(getTensor(srcDesc).dtype()))
			{
				case 1:
					kernel_im2col_2d(getConvolution(config), getTensor(colDesc), getPointer<int8_t>(colMem), getTensor(srcDesc),
							getPointer<int8_t>(srcMem), getTensor(filterDesc));
					break;
				case 2:
					kernel_im2col_2d(getConvolution(config), getTensor(colDesc), getPointer<int16_t>(colMem), getTensor(srcDesc),
							getPointer<int16_t>(srcMem), getTensor(filterDesc));
					break;
				case 4:
					kernel_im2col_2d(getConvolution(config), getTensor(colDesc), getPointer<int32_t>(colMem), getTensor(srcDesc),
							getPointer<int32_t>(srcMem), getTensor(filterDesc));
					break;
				case 8:
					kernel_im2col_2d(getConvolution(config), getTensor(colDesc), getPointer<int2>(colMem), getTensor(srcDesc),
							getPointer<int2>(srcMem), getTensor(filterDesc));
					break;
				case 16:
					kernel_im2col_2d(getConvolution(config), getTensor(colDesc), getPointer<int4>(colMem), getTensor(srcDesc),
							getPointer<int4>(srcMem), getTensor(filterDesc));
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refGetConvolutionWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config,
				const avTensorDescriptor_t xDesc, const avTensorDescriptor_t wDesc, const avTensorDescriptor_t bDesc, int *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			result[0] = 0;
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refPrecomputeConvolutionWorkspace(avContextDescriptor_t context, const avConvolutionDescriptor_t config,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				avMemoryDescriptor_t workspace)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation, avMemoryDescriptor_t workspace)
		{
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
					kernel_convolution_2d<int8_t, int32_t, float, float>(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc),
							getPointer<int8_t>(xMem), getTensor(wDesc), getPointer<int8_t>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<int8_t>(yMem), activation, getAlphaValue(alpha2), getPointer<float>(bMem), getPointer<int8_t>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_convolution_2d<float16, float, float>(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc),
							getPointer<float16>(xMem), getTensor(wDesc), getPointer<float16>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<float16>(yMem), activation, getAlphaValue(alpha2), getPointer<float16>(bMem), getPointer<float16>(zMem));
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_convolution_2d<bfloat16, float, float>(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc),
							getPointer<bfloat16>(xMem), getTensor(wDesc), getPointer<bfloat16>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<bfloat16>(yMem), activation, getAlphaValue(alpha2), getPointer<bfloat16>(bMem), getPointer<bfloat16>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_convolution_2d(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc), getPointer<float>(xMem), getTensor(wDesc),
							getPointer<float>(wMem), getBetaValue(beta), getTensor(yDesc), getPointer<float>(yMem), activation, getAlphaValue(alpha2),
							getPointer<float>(bMem), getPointer<float>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_convolution_2d(getConvolution(config), getAlphaValue<double>(alpha1), getTensor(xDesc), getPointer<double>(xMem),
							getTensor(wDesc), getPointer<double>(wMem), getBetaValue<double>(beta), getTensor(yDesc), getPointer<double>(yMem),
							activation, getAlphaValue<double>(alpha2), getPointer<double>(bMem), getPointer<double>(zMem));
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refConvolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
					kernel_convolution_2d<int8_t, int32_t, float>(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc),
							getPointer<int8_t>(xMem), getTensor(wDesc), getPointer<int8_t>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<int8_t>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_convolution_2d<float16, float, float>(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc),
							getPointer<float16>(xMem), getTensor(wDesc), getPointer<float16>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<float16>(yMem));
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_convolution_2d<bfloat16, float, float>(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc),
							getPointer<bfloat16>(xMem), getTensor(wDesc), getPointer<bfloat16>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<bfloat16>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_convolution_2d(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc), getPointer<float>(xMem), getTensor(wDesc),
							getPointer<float>(wMem), getBetaValue(beta), getTensor(yDesc), getPointer<float>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_convolution_2d(getConvolution(config), getAlphaValue<double>(alpha), getTensor(xDesc), getPointer<double>(xMem),
							getTensor(wDesc), getPointer<double>(wMem), getBetaValue<double>(beta), getTensor(yDesc), getPointer<double>(yMem));
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem)
		{
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_convolution_2d_update(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc), getPointer<float>(xMem),
							getTensor(dwDesc), getPointer<float>(dwMem), getBetaValue(beta), getTensor(dyDesc), getPointer<float>(dyMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_convolution_2d_update(getConvolution(config), getAlphaValue<double>(alpha), getTensor(xDesc), getPointer<double>(xMem),
							getTensor(dwDesc), getPointer<double>(dwMem), getBetaValue<double>(beta), getTensor(dyDesc), getPointer<double>(dyMem));
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

