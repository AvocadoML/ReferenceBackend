/*
 * convolution.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

#include "utils.hpp"
#include "activations.hpp"

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::reference;

	template<typename DataType, typename ComputeType = DataType, typename ScalingType = DataType>
	void kernel_convolution_1d(const ConvolutionDescriptor &config, ScalingType alpha1, const TensorDescriptor &xDesc, const DataType *xMem,
			const TensorDescriptor &wDesc, const DataType *wMem, ScalingType beta, const TensorDescriptor &yDesc, DataType *yMem,
			avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR, ScalingType alpha2 = zero<ScalingType>(), const ScalingType *bMem = nullptr,
			const DataType *zMem = nullptr)
	{
		const int batch_size = xDesc.dimension(0);

		const int output_filters = wDesc.firstDim();
		const int filter_height = wDesc.dimension(1);
		const int input_filters = wDesc.lastDim();

		const int padding_h = config.padding[0];
		const int stride_h = config.stride[0];
		const int dilation_h = config.dilation[0];

		const DataType padding_value = getScalarValue<DataType>(config.padding_value.data());

		if (beta == zero<ScalingType>())
			clear(yMem, yDesc.volume());

		for (int b = 0; b < batch_size; b++) // batch size
			for (int g = 0; g < config.groups; g++)
			{
				const int output_filters_group[2] = { g * output_filters / config.groups, (g + 1) * output_filters / config.groups };
				const int input_filters_group[2] = { g * input_filters / config.groups, (g + 1) * input_filters / config.groups };

				for (int out = output_filters_group[0]; out < output_filters_group[1]; out++) // output filters
					for (int out_h = 0; out_h < yDesc.dimension(1); out_h++) // output height
					{
						ComputeType tmp = zero<ComputeType>();
						for (int i = 0; i < wDesc.dimension(1); i++) // kernel height
						{
							int x;
							if (config.mode == AVOCADO_CONVOLUTION_MODE)
								x = padding_h + i * dilation_h + out_h * stride_h;
							else
								// AVOCADO_CROSS_CORRELATION_MODE
								x = padding_h + (filter_height - 1 - i) * dilation_h + out_h * stride_h;
							if (x >= 0 and x < xDesc.dimension(1))
							{
								for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
									tmp += static_cast<ComputeType>(wMem[wDesc.getIndex( { out, i, in })])
											* static_cast<ComputeType>(xMem[xDesc.getIndex( { b, x, in })]);
							}
							else
							{
								for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
									tmp += static_cast<ComputeType>(wMem[wDesc.getIndex( { out, i, in })]) * static_cast<ComputeType>(padding_value);
							}
						}
						ScalingType tmp2 = alpha1 * static_cast<ScalingType>(tmp) + beta * static_cast<ScalingType>(yMem[yDesc.getIndex( { b, out_h,
								out })]);
						if (bMem != nullptr)
							tmp2 += static_cast<ScalingType>(bMem[out]);
						if (zMem != nullptr)
							tmp2 += alpha2 * static_cast<ScalingType>(zMem[yDesc.getIndex( { b, out_h, out })]);
						tmp2 = activation_forward(activation, tmp2);
						yMem[yDesc.getIndex( { b, out_h, out })] = Store<DataType, ScalingType>::store(tmp2);
					}
			}
	}
	template<typename DataType, typename ComputeType = DataType, typename ScalingType = DataType>
	void kernel_convolution_2d(const ConvolutionDescriptor &config, ScalingType alpha1, const TensorDescriptor &xDesc, const DataType *xMem,
			const TensorDescriptor &wDesc, const DataType *wMem, ScalingType beta, const TensorDescriptor &yDesc, DataType *yMem,
			avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR, ScalingType alpha2 = zero<ScalingType>(), const ScalingType *bMem = nullptr,
			const DataType *zMem = nullptr)
	{
		const int batch_size = xDesc.dimension(0);

		const int output_filters = wDesc.firstDim();
		const int filter_height = wDesc.dimension(1);
		const int filter_width = wDesc.dimension(2);
		const int input_filters = wDesc.lastDim();

		const int padding_h = config.padding[0];
		const int padding_w = config.padding[1];

		const int stride_h = config.stride[0];
		const int stride_w = config.stride[1];

		const int dilation_h = config.dilation[0];
		const int dilation_w = config.dilation[1];

		const DataType padding_value = getScalarValue<DataType>(config.padding_value.data());

		if (beta == zero<ScalingType>())
			clear(yMem, yDesc.volume());

		for (int b = 0; b < batch_size; b++) // batch size
			for (int g = 0; g < config.groups; g++)
			{
				const int output_filters_group[2] = { g * output_filters / config.groups, (g + 1) * output_filters / config.groups };
				const int input_filters_group[2] = { g * input_filters / config.groups, (g + 1) * input_filters / config.groups };

				for (int out_f = output_filters_group[0]; out_f < output_filters_group[1]; out_f++) // output filters
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
										for (int in_f = input_filters_group[0]; in_f < input_filters_group[1]; in_f++) // input filters
											tmp += static_cast<ComputeType>(wMem[wDesc.getIndex( { out_f, i, j, in_f })])
													* static_cast<ComputeType>(xMem[xDesc.getIndex( { b, x, y, in_f })]);
									}
									else
									{
										for (int in_f = input_filters_group[0]; in_f < input_filters_group[1]; in_f++) // input filters
											tmp += static_cast<ComputeType>(wMem[wDesc.getIndex( { out_f, i, j, in_f })])
													* static_cast<ComputeType>(padding_value);
									}
								}
							ScalingType tmp2 = alpha1 * static_cast<ScalingType>(tmp) + beta * static_cast<ScalingType>(yMem[yDesc.getIndex( { b,
									out_h, out_w, out_f })]);
							if (bMem != nullptr)
								tmp2 += static_cast<ScalingType>(bMem[out_f]);
							if (zMem != nullptr)
								tmp2 += alpha2 * static_cast<ScalingType>(zMem[yDesc.getIndex( { b, out_h, out_w, out_f })]);
							tmp2 = activation_forward(activation, tmp2);
							yMem[yDesc.getIndex( { b, out_h, out_w, out_f })] = Store<DataType, ScalingType>::store(tmp2);
						}
			}
	}

	template<typename T>
	void kernel_convolution_backward_1d(const ConvolutionDescriptor &config, T alpha, const TensorDescriptor &dxDesc, T *dxMem,
			const TensorDescriptor &wDesc, const T *wMem, T beta, const TensorDescriptor &dyDesc, const T *dyMem)
	{
		const int batch_size = dxDesc.dimension(0);

		const int output_filters = wDesc.firstDim();
		const int filter_height = wDesc.dimension(1);
		const int input_filters = wDesc.lastDim();

		const int padding_h = config.padding[0];
		const int stride_h = config.stride[0];
		const int dilation_h = config.dilation[0];

		if (beta == zero<T>())
			clear(dxMem, dxDesc.volume());
		else
		{
			for (int i = 0; i < dxDesc.volume(); i++)
				dxMem[i] *= beta;
		}

		for (int b = 0; b < batch_size; b++) // batch size
			for (int g = 0; g < config.groups; g++)
			{
				const int output_filters_group[2] = { g * output_filters / config.groups, (g + 1) * output_filters / config.groups };
				const int input_filters_group[2] = { g * input_filters / config.groups, (g + 1) * input_filters / config.groups };

				for (int out_f = output_filters_group[0]; out_f < output_filters_group[1]; out_f++) // output filters
					for (int out_h = 0; out_h < dyDesc.dimension(1); out_h++) // output height
					{
						for (int i = 0; i < wDesc.dimension(1); i++) // kernel height
						{
							int x;
							if (config.mode == AVOCADO_CONVOLUTION_MODE)
								x = padding_h + i * dilation_h + out_h * stride_h;
							else
								// AVOCADO_CROSS_CORRELATION_MODE
								x = padding_h + (filter_height - 1 - i) * dilation_h + out_h * stride_h;

							if (x >= 0 and x < dxDesc.dimension(1))
							{
								for (int in_f = input_filters_group[0]; in_f < input_filters_group[1]; in_f++) // input filters
								{
									T tmp = wMem[wDesc.getIndex( { out_f, i, in_f })] * dyMem[dyDesc.getIndex( { b, out_h, out_f })];
									dxMem[dxDesc.getIndex( { b, x, in_f })] += alpha * tmp;
								}
							}
						}
					}
			}
	}
	template<typename T>
	void kernel_convolution_backward_2d(const ConvolutionDescriptor &config, T alpha, const TensorDescriptor &dxDesc, T *dxMem,
			const TensorDescriptor &wDesc, const T *wMem, T beta, const TensorDescriptor &dyDesc, const T *dyMem)
	{
		const int batch_size = dxDesc.dimension(0);

		const int output_filters = wDesc.firstDim();
		const int filter_height = wDesc.dimension(1);
		const int filter_width = wDesc.dimension(2);
		const int input_filters = wDesc.lastDim();

		const int padding_h = config.padding[0];
		const int padding_w = config.padding[1];

		const int stride_h = config.stride[0];
		const int stride_w = config.stride[1];

		const int dilation_h = config.dilation[0];
		const int dilation_w = config.dilation[1];

		if (beta == zero<T>())
			clear(dxMem, dxDesc.volume());
		else
		{
			for (int i = 0; i < dxDesc.volume(); i++)
				dxMem[i] *= beta;
		}

		for (int b = 0; b < batch_size; b++) // batch size
			for (int g = 0; g < config.groups; g++)
			{
				const int output_filters_group[2] = { g * output_filters / config.groups, (g + 1) * output_filters / config.groups };
				const int input_filters_group[2] = { g * input_filters / config.groups, (g + 1) * input_filters / config.groups };

				for (int out_f = output_filters_group[0]; out_f < output_filters_group[1]; out_f++) // output filters
					for (int out_h = 0; out_h < dyDesc.dimension(1); out_h++) // output height
						for (int out_w = 0; out_w < dyDesc.dimension(2); out_w++) // output width
						{
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
									if (x >= 0 and x < dxDesc.dimension(1) and y >= 0 and y < dxDesc.dimension(2))
									{
										for (int in_f = input_filters_group[0]; in_f < input_filters_group[1]; in_f++) // input filters
										{
											T tmp = wMem[wDesc.getIndex( { out_f, i, j, in_f })]
													* dyMem[dyDesc.getIndex( { b, out_h, out_w, out_f })];
											dxMem[dxDesc.getIndex( { b, x, y, in_f })] += alpha * tmp;
										}
									}
								}
						}
			}
	}

	template<typename T>
	void kernel_convolution_1d_update(const ConvolutionDescriptor &config, T alpha, const TensorDescriptor &xDesc, const T *xMem,
			const TensorDescriptor &dwDesc, T *dwMem, T beta, const TensorDescriptor &dyDesc, const T *dyMem)
	{
		const int batch_size = xDesc.dimension(0);

		const int output_filters = dwDesc.firstDim();
		const int filter_height = dwDesc.dimension(1);
		const int input_filters = dwDesc.lastDim();

		const int padding_h = config.padding[0];
		const int stride_h = config.stride[0];
		const int dilation_h = config.dilation[0];

		const T padding_value = getScalarValue<T>(config.padding_value.data());

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

				for (int out_f = output_filters_group[0]; out_f < output_filters_group[1]; out_f++) // output filters
					for (int out_h = 0; out_h < dyDesc.dimension(1); out_h++) // output height
					{
						for (int i = 0; i < dwDesc.dimension(1); i++) // kernel height
						{
							int x;
							if (config.mode == AVOCADO_CONVOLUTION_MODE)
								x = padding_h + i * dilation_h + out_h * stride_h;
							else
								x = padding_h + (filter_height - 1 - i) * dilation_h + out_h * stride_h;
							if (x >= 0 and x < xDesc.dimension(1))
							{
								for (int in_f = input_filters_group[0]; in_f < input_filters_group[1]; in_f++) // input filters
									dwMem[dwDesc.getIndex( { out_f, i, in_f })] += alpha * dyMem[dyDesc.getIndex( { out_f, out_h, in_f })]
											* xMem[xDesc.getIndex( { b, x, in_f })];
							}
							else
							{
								for (int in_f = input_filters_group[0]; in_f < input_filters_group[1]; in_f++) // input filters
									dwMem[dwDesc.getIndex( { out_f, i, in_f })] += alpha * dyMem[dyDesc.getIndex( { b, out_h, out_f })]
											* padding_value;
							}
						}
					}
			}
	}
	template<typename T>
	void kernel_convolution_2d_update(const ConvolutionDescriptor &config, T alpha, const TensorDescriptor &xDesc, const T *xMem,
			const TensorDescriptor &dwDesc, T *dwMem, T beta, const TensorDescriptor &dyDesc, const T *dyMem)
	{
		const int batch_size = xDesc.dimension(0);

		const int output_filters = dwDesc.firstDim();
		const int filter_height = dwDesc.dimension(1);
		const int filter_width = dwDesc.dimension(2);
		const int input_filters = dwDesc.lastDim();

		const int padding_h = config.padding[0];
		const int padding_w = config.padding[1];

		const int stride_h = config.stride[0];
		const int stride_w = config.stride[1];

		const int dilation_h = config.dilation[0];
		const int dilation_w = config.dilation[1];

		const T padding_value = getScalarValue<T>(config.padding_value.data());

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

				for (int out_f = output_filters_group[0]; out_f < output_filters_group[1]; out_f++) // output filters
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
										for (int in_f = input_filters_group[0]; in_f < input_filters_group[1]; in_f++) // input filters
											dwMem[dwDesc.getIndex( { out_f, i, j, in_f })] += alpha
													* dyMem[dyDesc.getIndex( { b, out_h, out_w, out_f })] * xMem[xDesc.getIndex( { b, x, y, in_f })];
									}
									else
									{
										for (int in_f = input_filters_group[0]; in_f < input_filters_group[1]; in_f++) // input filters
											dwMem[dwDesc.getIndex( { out_f, i, j, in_f })] += alpha
													* dyMem[dyDesc.getIndex( { b, out_h, out_w, out_f })] * padding_value;
									}
								}
						}
			}
	}

	template<typename DataType, typename ComputeType = DataType, typename ScalingType = DataType>
	avStatus_t launcher_convolution_forward(const ConvolutionDescriptor &config, ScalingType alpha1, const TensorDescriptor &xDesc,
			const DataType *xMem, const TensorDescriptor &wDesc, const DataType *wMem, ScalingType beta, const TensorDescriptor &yDesc,
			DataType *yMem, avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR, ScalingType alpha2 = zero<ScalingType>(),
			const ScalingType *bMem = nullptr, const DataType *zMem = nullptr)
	{
		switch (wDesc.nbDims())
		{
			case 3: // 1D convolution
				kernel_convolution_1d(config, alpha1, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem, activation, alpha2, bMem, zMem);
				return AVOCADO_STATUS_NOT_SUPPORTED;
			case 4: // 2D convolution
				kernel_convolution_2d(config, alpha1, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem, activation, alpha2, bMem, zMem);
				return AVOCADO_STATUS_SUCCESS;
			case 5: // 3D convolution
				return AVOCADO_STATUS_NOT_SUPPORTED; // TODO
			default:
				return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	}
	template<typename T>
	avStatus_t launcher_convolution_backward(const ConvolutionDescriptor &config, T alpha, const TensorDescriptor &dxDesc, T *dxMem,
			const TensorDescriptor &wDesc, const T *wMem, T beta, const TensorDescriptor &dyDesc, const T *dyMem)
	{
		switch (wDesc.nbDims())
		{
			case 3: // 1D convolution
				kernel_convolution_backward_1d(config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem);
				return AVOCADO_STATUS_NOT_SUPPORTED;
			case 4: // 2D convolution
				kernel_convolution_backward_2d(config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem);
				return AVOCADO_STATUS_SUCCESS;
			case 5: // 3D convolution
				return AVOCADO_STATUS_NOT_SUPPORTED; // TODO
			default:
				return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	}
	template<typename T>
	avStatus_t launcher_convolution_update(const ConvolutionDescriptor &config, T alpha, const TensorDescriptor &xDesc, const T *xMem,
			const TensorDescriptor &dwDesc, T *dwMem, T beta, const TensorDescriptor &dyDesc, const T *dyMem)
	{
		switch (dwDesc.nbDims())
		{
			case 3: // 1D convolution
				kernel_convolution_1d_update(config, alpha, xDesc, xMem, dwDesc, dwMem, beta, dyDesc, dyMem);
				return AVOCADO_STATUS_NOT_SUPPORTED;
			case 4: // 2D convolution
				kernel_convolution_2d_update(config, alpha, xDesc, xMem, dwDesc, dwMem, beta, dyDesc, dyMem);
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
		using namespace reference;
		avStatus_t refGetConvolutionWorkspaceSize(const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avTensorDescriptor_t wDesc, bool inferenceOnly, avSize_t *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			result[0] = 0;
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation, avMemoryDescriptor_t workspaceMem)
		{
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
					launcher_convolution_forward<int8_t, int32_t, float>(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc),
							getPointer<int8_t>(xMem), getTensor(wDesc), getPointer<int8_t>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<int8_t>(yMem), activation, getAlphaValue(alpha2), getPointer<float>(bMem), getPointer<int8_t>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT16:
					launcher_convolution_forward<float16, float, float>(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc),
							getPointer<float16>(xMem), getTensor(wDesc), getPointer<float16>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<float16>(yMem), activation, getAlphaValue(alpha2), getPointer<float>(bMem), getPointer<float16>(zMem));
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					launcher_convolution_forward<bfloat16, float, float>(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc),
							getPointer<bfloat16>(xMem), getTensor(wDesc), getPointer<bfloat16>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<bfloat16>(yMem), activation, getAlphaValue(alpha2), getPointer<float>(bMem), getPointer<bfloat16>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT32:
					launcher_convolution_forward(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc), getPointer<float>(xMem),
							getTensor(wDesc), getPointer<float>(wMem), getBetaValue(beta), getTensor(yDesc), getPointer<float>(yMem), activation,
							getAlphaValue(alpha2), getPointer<float>(bMem), getPointer<float>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_convolution_forward(getConvolution(config), getAlphaValue<double>(alpha1), getTensor(xDesc), getPointer<double>(xMem),
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
				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t workspaceMem)
		{
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
					launcher_convolution_forward<int8_t, int32_t, float>(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc),
							getPointer<int8_t>(xMem), getTensor(wDesc), getPointer<int8_t>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<int8_t>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT16:
					launcher_convolution_forward<float16, float, float>(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc),
							getPointer<float16>(xMem), getTensor(wDesc), getPointer<float16>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<float16>(yMem));
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					launcher_convolution_forward<bfloat16, float, float>(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc),
							getPointer<bfloat16>(xMem), getTensor(wDesc), getPointer<bfloat16>(wMem), getBetaValue(beta), getTensor(yDesc),
							getPointer<bfloat16>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT32:
					launcher_convolution_forward(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc), getPointer<float>(xMem),
							getTensor(wDesc), getPointer<float>(wMem), getBetaValue(beta), getTensor(yDesc), getPointer<float>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_convolution_forward(getConvolution(config), getAlphaValue<double>(alpha), getTensor(xDesc), getPointer<double>(xMem),
							getTensor(wDesc), getPointer<double>(wMem), getBetaValue<double>(beta), getTensor(yDesc), getPointer<double>(yMem));
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refConvolutionBackward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const void *beta, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, avMemoryDescriptor_t workspaceMem)
		{
			switch (getTensor(dxDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					launcher_convolution_backward(getConvolution(config), getAlphaValue(alpha), getTensor(dxDesc), getPointer<float>(dxMem),
							getTensor(wDesc), getPointer<float>(wMem), getBetaValue(beta), getTensor(dyDesc), getPointer<float>(dyMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_convolution_backward(getConvolution(config), getAlphaValue<double>(alpha), getTensor(dxDesc), getPointer<double>(dxMem),
							getTensor(wDesc), getPointer<double>(wMem), getBetaValue<double>(beta), getTensor(dyDesc), getPointer<double>(dyMem));
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem,
				avMemoryDescriptor_t workspaceMem)
		{
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					launcher_convolution_update(getConvolution(config), getAlphaValue(alpha), getTensor(xDesc), getPointer<float>(xMem),
							getTensor(dwDesc), getPointer<float>(dwMem), getBetaValue(beta), getTensor(dyDesc), getPointer<float>(dyMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_convolution_update(getConvolution(config), getAlphaValue<double>(alpha), getTensor(xDesc), getPointer<double>(xMem),
							getTensor(dwDesc), getPointer<double>(dwMem), getBetaValue<double>(beta), getTensor(dyDesc), getPointer<double>(dyMem));
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

