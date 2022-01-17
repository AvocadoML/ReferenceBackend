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

	struct int2
	{
			int x0, x1;
	};

	struct int4
	{
			int x0, x1, x2, x3;
	};

	template<typename DataType, typename ComputeType = DataType, typename ScalingType = DataType>
	void kernel_convolution_1d(const reference::ConvolutionDescriptor &config, ScalingType alpha1, const reference::TensorDescriptor &xDesc,
			const DataType *xMem, const reference::TensorDescriptor &wDesc, const DataType *wMem, ScalingType beta,
			const reference::TensorDescriptor &yDesc, DataType *yMem, avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR, ScalingType alpha2 =
					zero<ScalingType>(), const ScalingType *bMem = nullptr, const DataType *zMem = nullptr)
	{
		const int batch_size = xDesc.dimension(0);

		const int output_filters = wDesc.firstDim();
		const int filter_height = wDesc.dimension(1);
		const int input_filters = wDesc.lastDim();

		const int padding_h = config.padding[0];
		const int stride_h = config.stride[0];
		const int dilation_h = config.dilation[0];

		const DataType padding_value = reference::getScalarValue<DataType>(config.padding_value.data());

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
	void kernel_convolution_2d(const reference::ConvolutionDescriptor &config, ScalingType alpha1, const reference::TensorDescriptor &xDesc,
			const DataType *xMem, const reference::TensorDescriptor &wDesc, const DataType *wMem, ScalingType beta,
			const reference::TensorDescriptor &yDesc, DataType *yMem, avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR, ScalingType alpha2 =
					zero<ScalingType>(), const ScalingType *bMem = nullptr, const DataType *zMem = nullptr)
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

		const DataType padding_value = reference::getScalarValue<DataType>(config.padding_value.data());

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
							ScalingType tmp2 = alpha1 * static_cast<ScalingType>(tmp) + beta * static_cast<ScalingType>(yMem[yDesc.getIndex( { b,
									out_h, out_w, out })]);
							if (bMem != nullptr)
								tmp2 += static_cast<ScalingType>(bMem[out]);
							if (zMem != nullptr)
								tmp2 += alpha2 * static_cast<ScalingType>(zMem[yDesc.getIndex( { b, out_h, out_w, out })]);
							tmp2 = activation_forward(activation, tmp2);
							yMem[yDesc.getIndex( { b, out_h, out_w, out })] = Store<DataType, ScalingType>::store(tmp2);
						}
			}
	}

	template<typename T>
	void kernel_convolution_1d_update(const reference::ConvolutionDescriptor &config, T alpha, const reference::TensorDescriptor &xDesc,
			const T *xMem, const reference::TensorDescriptor &dwDesc, T *dwMem, T beta, const reference::TensorDescriptor &dyDesc, const T *dyMem)
	{
		const int batch_size = xDesc.dimension(0);

		const int output_filters = dwDesc.firstDim();
		const int filter_height = dwDesc.dimension(1);
		const int input_filters = dwDesc.lastDim();

		const int padding_h = config.padding[0];
		const int stride_h = config.stride[0];
		const int dilation_h = config.dilation[0];

		const T padding_value = reference::getScalarValue<T>(config.padding_value.data());

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
								for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
									dwMem[dwDesc.getIndex( { out, i, in })] += alpha * dyMem[dyDesc.getIndex( { out, out_h, in })]
											* xMem[xDesc.getIndex( { b, x, in })];
							}
							else
							{
								for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
									dwMem[dwDesc.getIndex( { out, i, in })] += alpha * dyMem[dyDesc.getIndex( { out, out_h, in })] * padding_value;
							}
						}
					}
			}
	}
	template<typename T>
	void kernel_convolution_2d_update(const reference::ConvolutionDescriptor &config, T alpha, const reference::TensorDescriptor &xDesc,
			const T *xMem, const reference::TensorDescriptor &dwDesc, T *dwMem, T beta, const reference::TensorDescriptor &dyDesc, const T *dyMem)
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

		const T padding_value = reference::getScalarValue<T>(config.padding_value.data());

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

	template<typename DataType, typename ComputeType = DataType, typename ScalingType = DataType>
	avStatus_t launcher_convolution_forward(const reference::ConvolutionDescriptor &config, ScalingType alpha1,
			const reference::TensorDescriptor &xDesc, const DataType *xMem, const reference::TensorDescriptor &wDesc, const DataType *wMem,
			ScalingType beta, const reference::TensorDescriptor &yDesc, DataType *yMem, avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR,
			ScalingType alpha2 = zero<ScalingType>(), const ScalingType *bMem = nullptr, const DataType *zMem = nullptr)
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
	avStatus_t launcher_convolution_update(const reference::ConvolutionDescriptor &config, T alpha, const reference::TensorDescriptor &xDesc,
			const T *xMem, const reference::TensorDescriptor &dwDesc, T *dwMem, T beta, const reference::TensorDescriptor &dyDesc, const T *dyMem)
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

		avStatus_t refGetConvolutionWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config,
				const avTensorDescriptor_t xDesc, const avTensorDescriptor_t wDesc, const avTensorDescriptor_t bDesc, int *result)
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
			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
					launcher_convolution_forward<int8_t, int32_t, float>(reference::getConvolution(config), reference::getAlphaValue(alpha1),
							reference::getTensor(xDesc), reference::getPointer<int8_t>(xMem), reference::getTensor(wDesc),
							reference::getPointer<int8_t>(wMem), reference::getBetaValue(beta), reference::getTensor(yDesc),
							reference::getPointer<int8_t>(yMem), activation, reference::getAlphaValue(alpha2), reference::getPointer<float>(bMem),
							reference::getPointer<int8_t>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT16:
					launcher_convolution_forward<float16, float, float>(reference::getConvolution(config), reference::getAlphaValue(alpha1),
							reference::getTensor(xDesc), reference::getPointer<float16>(xMem), reference::getTensor(wDesc),
							reference::getPointer<float16>(wMem), reference::getBetaValue(beta), reference::getTensor(yDesc),
							reference::getPointer<float16>(yMem), activation, reference::getAlphaValue(alpha2), reference::getPointer<float>(bMem),
							reference::getPointer<float16>(zMem));
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					launcher_convolution_forward<bfloat16, float, float>(reference::getConvolution(config), reference::getAlphaValue(alpha1),
							reference::getTensor(xDesc), reference::getPointer<bfloat16>(xMem), reference::getTensor(wDesc),
							reference::getPointer<bfloat16>(wMem), reference::getBetaValue(beta), reference::getTensor(yDesc),
							reference::getPointer<bfloat16>(yMem), activation, reference::getAlphaValue(alpha2), reference::getPointer<float>(bMem),
							reference::getPointer<bfloat16>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT32:
					launcher_convolution_forward(reference::getConvolution(config), reference::getAlphaValue(alpha1), reference::getTensor(xDesc),
							reference::getPointer<float>(xMem), reference::getTensor(wDesc), reference::getPointer<float>(wMem),
							reference::getBetaValue(beta), reference::getTensor(yDesc), reference::getPointer<float>(yMem), activation,
							reference::getAlphaValue(alpha2), reference::getPointer<float>(bMem), reference::getPointer<float>(zMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_convolution_forward(reference::getConvolution(config), reference::getAlphaValue<double>(alpha1),
							reference::getTensor(xDesc), reference::getPointer<double>(xMem), reference::getTensor(wDesc),
							reference::getPointer<double>(wMem), reference::getBetaValue<double>(beta), reference::getTensor(yDesc),
							reference::getPointer<double>(yMem), activation, reference::getAlphaValue<double>(alpha2),
							reference::getPointer<double>(bMem), reference::getPointer<double>(zMem));
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
			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
					launcher_convolution_forward<int8_t, int32_t, float>(reference::getConvolution(config), reference::getAlphaValue(alpha),
							reference::getTensor(xDesc), reference::getPointer<int8_t>(xMem), reference::getTensor(wDesc),
							reference::getPointer<int8_t>(wMem), reference::getBetaValue(beta), reference::getTensor(yDesc),
							reference::getPointer<int8_t>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT16:
					launcher_convolution_forward<float16, float, float>(reference::getConvolution(config), reference::getAlphaValue(alpha),
							reference::getTensor(xDesc), reference::getPointer<float16>(xMem), reference::getTensor(wDesc),
							reference::getPointer<float16>(wMem), reference::getBetaValue(beta), reference::getTensor(yDesc),
							reference::getPointer<float16>(yMem));
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					launcher_convolution_forward<bfloat16, float, float>(reference::getConvolution(config), reference::getAlphaValue(alpha),
							reference::getTensor(xDesc), reference::getPointer<bfloat16>(xMem), reference::getTensor(wDesc),
							reference::getPointer<bfloat16>(wMem), reference::getBetaValue(beta), reference::getTensor(yDesc),
							reference::getPointer<bfloat16>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT32:
					launcher_convolution_forward(reference::getConvolution(config), reference::getAlphaValue(alpha), reference::getTensor(xDesc),
							reference::getPointer<float>(xMem), reference::getTensor(wDesc), reference::getPointer<float>(wMem),
							reference::getBetaValue(beta), reference::getTensor(yDesc), reference::getPointer<float>(yMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_convolution_forward(reference::getConvolution(config), reference::getAlphaValue<double>(alpha),
							reference::getTensor(xDesc), reference::getPointer<double>(xMem), reference::getTensor(wDesc),
							reference::getPointer<double>(wMem), reference::getBetaValue<double>(beta), reference::getTensor(yDesc),
							reference::getPointer<double>(yMem));
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
			reference::ConvolutionDescriptor cfg = reference::getConvolution(config);
			if (cfg.mode == AVOCADO_CONVOLUTION_MODE)
				cfg.mode = AVOCADO_CROSS_CORRELATION_MODE;
			else
				cfg.mode = AVOCADO_CONVOLUTION_MODE;

			for (int i = 0; i < reference::getTensor(wDesc).nbDims() - 2; i++)
				cfg.padding[i] = -(reference::getTensor(wDesc).dimension(1 + i) - 1) - cfg.padding[i];

			switch (reference::getTensor(dxDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					launcher_convolution_forward(cfg, reference::getAlphaValue(alpha), reference::getTensor(dyDesc),
							reference::getPointer<float>(dyMem), reference::getTensor(wDesc), reference::getPointer<float>(wMem),
							reference::getBetaValue(beta), reference::getTensor(dxDesc), reference::getPointer<float>(dxMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_convolution_forward(cfg, reference::getAlphaValue<double>(alpha), reference::getTensor(dyDesc),
							reference::getPointer<double>(dyMem), reference::getTensor(wDesc), reference::getPointer<double>(wMem),
							reference::getBetaValue<double>(beta), reference::getTensor(dxDesc), reference::getPointer<double>(dxMem));
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
			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					launcher_convolution_update(reference::getConvolution(config), reference::getAlphaValue(alpha), reference::getTensor(xDesc),
							reference::getPointer<float>(xMem), reference::getTensor(dwDesc), reference::getPointer<float>(dwMem),
							reference::getBetaValue(beta), reference::getTensor(dyDesc), reference::getPointer<float>(dyMem));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_convolution_update(reference::getConvolution(config), reference::getAlphaValue<double>(alpha),
							reference::getTensor(xDesc), reference::getPointer<double>(xMem), reference::getTensor(dwDesc),
							reference::getPointer<double>(dwMem), reference::getBetaValue<double>(beta), reference::getTensor(dyDesc),
							reference::getPointer<double>(dyMem));
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

