/*
 * convolution.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "utils.hpp"

namespace
{
	struct int2
	{
			int x0, x1;
	};

	struct int4
	{
			int x0, x1, x2, x3;
	};

//	void baseline_conv2d_forward(const math::ConvConfig &config, const Tensor &input, Tensor &output, const Tensor &weight, const Tensor &bias,
//			const Tensor &add, NonlinearityType act)
//	{
//		assert(same_device(input, output, weight, bias));
//		if (not add.isEmpty())
//		{
//			assert(same_device(output, add));
//			assert(same_shape(output, add));
//		}
//		assert(input.device().isCPU());
//		assert(input.lastDim() == weight.lastDim()); // input filters
//		assert(output.lastDim() == weight.firstDim()); // output filters
//
//		output.zeroall();
//		for (int b = 0; b < input.shape(0); b++) // batch size
//			for (int g = 0; g < config.groups; g++)
//			{
//				const int output_filters[2] = { g * output.shape(3) / config.groups, (g + 1) * output.shape(3) / config.groups };
//				const int input_filters[2] = { g * input.shape(3) / config.groups, (g + 1) * input.shape(3) / config.groups };
//				for (int out = output_filters[0]; out < output_filters[1]; out++) // output filters
//					for (int out_h = 0; out_h < output.shape(1); out_h++) // output height
//						for (int out_w = 0; out_w < output.shape(2); out_w++) // output width
//						{
//							float tmp = 0.0f;
//							for (int i = 0; i < weight.shape(1); i++) // kernel height
//								for (int j = 0; j < weight.shape(2); j++) // kernel width
//								{
//									int x = config.padding[0] + out_h * config.stride[0] + i * config.dilation[0];
//									int y = config.padding[1] + out_w * config.stride[1] + j * config.dilation[1];
//									if (x >= 0 and x < input.shape(1) and y >= 0 and y < input.shape(2))
//										for (int in = input_filters[0]; in < input_filters[1]; in++) // input filters
//											tmp += weight.get<float>( { out, i, j, in }) * input.get<float>( { b, x, y, in });
//								}
//							if (not bias.isEmpty())
//								tmp += bias.get<float>( { out });
//							if (not add.isEmpty())
//								tmp += add.get<float>( { b, out_h, out_w, out });
//							output.set(tmp, { b, out_h, out_w, out });
//						}
//			}
//		math::nonlinearityForwardInPlace(Context(), output, act);
//	}
//	void baseline_conv2d_backward(const math::ConvConfig &config, const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next,
//			const Tensor &weight, NonlinearityType act)
//	{
//		assert(same_device(output, gradient_prev, gradient_next, weight));
//		assert(output.device().isCPU());
//		const int batch = output.shape(0);
//		const int height = output.shape(1);
//		const int width = output.shape(2);
//		const int filters_in = gradient_prev.shape(3);
//		const int filters_out = gradient_next.shape(3);
//
//		const int kernel_height = weight.shape(1);
//		const int kernel_width = weight.shape(2);
//
//		const int pad_h = -kernel_height / 2; //TODO handle padding
//		const int pad_w = -kernel_width / 2; //TODO handle padding
//
//		math::nonlinearityBackwardInPlace(Context(), gradient_next, output, act);
//		gradient_prev.zeroall();
//		for (int b = 0; b < batch; b++)
//			for (int out = 0; out < filters_out; out++)
//				for (int h = 0; h < height; h++)
//					for (int w = 0; w < width; w++)
//						for (int i = 0; i < kernel_height; i++)
//							for (int j = 0; j < kernel_width; j++)
//								if ((pad_h + h + i) >= 0 && (pad_h + h + i) < height && (pad_w + w + j) >= 0 && (pad_w + w + j) < width)
//									for (int in = 0; in < filters_in; in++)
//									{
//										float grad = gradient_next.get<float>( { b, h, w, out });
//										float we = weight.get<float>( { out, i, j, in });
//										float pr = gradient_prev.get<float>( { b, pad_h + h + i, pad_w + w + j, in });
//										gradient_prev.set(pr + grad * we, { b, pad_h + h + i, pad_w + w + j, in });
//									}
//	}
//	void baseline_conv2d_update(const Tensor &input, const Tensor &gradient_next, Tensor &weight_update)
//	{
//		assert(same_device(input, gradient_next, weight_update));
//		assert(input.device().isCPU());
//		const int batch = input.shape(0);
//		const int height = input.shape(1);
//		const int width = input.shape(2);
//		const int filters_in = input.shape(3);
//		const int filters_out = gradient_next.shape(3);
//
//		const int kernel_height = weight_update.shape(1);
//		const int kernel_width = weight_update.shape(2);
//
//		const int pad_h = -kernel_height / 2; //TODO handle padding
//		const int pad_w = -kernel_width / 2; //TODO handle padding
//
//		for (int b = 0; b < batch; b++)
//			for (int out = 0; out < filters_out; out++)
//			{
//				for (int in = 0; in < filters_in; in++)
//					for (int i = 0; i < kernel_height; i++)
//						for (int j = 0; j < kernel_width; j++)
//						{
//							float tmp = weight_update.get<float>( { out, i, j, in });
//							for (int h = 0; h < height; h++)
//								for (int w = 0; w < width; w++)
//									if ((pad_h + h + i) >= 0 && (pad_h + h + i) < height && (pad_w + w + j) >= 0 && (pad_w + w + j) < width)
//										tmp += gradient_next.get<float>( { b, h, w, out })
//												* input.get<float>( { b, pad_h + h + i, pad_w + w + j, in });
//							weight_update.set(tmp, { out, i, j, in });
//						}
//			}
//	}
//
//	template<typename T>
//	void im2col(const T *tensor, T *matrix, int2 kernel_size, int4 shape)
//	{
//		const int height = shape.x1;
//		const int width = shape.x2;
//		const int block_length = shape.x3 * sizeof(T);
//		const int pad_h = kernel_size.x0 / 2;
//		const int pad_w = kernel_size.x1 / 2;
//		//#pragma omp parallel
//		{
//			for (int b = 0; b < shape.x0; b++)
//			{
//				//#pragma omp for nowait
//				for (int h = 0; h < height; h++)
//					for (int w = 0; w < width; w++)
//					{
//						T *ptr_dst = matrix + ((b * height + h) * width + w) * shape.x3 * kernel_size.x * kernel_size.y; // start of a row
//						if (h >= pad_h && h < height - pad_h && w >= pad_w && w < width - pad_w) // center of the image
//						{
//							for (int i = -pad_h; i <= pad_h; i++, ptr_dst += kernel_size.y * shape.w)
//								std::memcpy(ptr_dst, tensor + ((b * height + h + i) * width + w - pad_w) * shape.w, kernel_size.y * block_length);
//						}
//						else // borders of the image
//						{
//							for (int i = -pad_h; i <= pad_h; i++)
//								for (int j = -pad_w; j <= pad_w; j++, ptr_dst += shape.w)
//									if ((h + i) >= 0 && (h + i) < height && (w + j) >= 0 && (w + j) < width)
//										std::memcpy(ptr_dst, tensor + ((b * height + h + i) * width + w + j) * shape.w, block_length);
//									else
//										std::memset(ptr_dst, 0, block_length);
//						}
//					}
//			}
//		}
//	}
//	template<typename T>
//	void im2col_inv(const T *tensor, T *matrix, int2 kernel_size, int4 shape)
//	{
//		const int height = shape.y;
//		const int width = shape.z;
//		const int block_length = shape.w * sizeof(T);
//		const int pad_h = kernel_size.x / 2;
//		const int pad_w = kernel_size.y / 2;
//#pragma omp parallel
//		{
//			for (int b = 0; b < shape.x; b++)
//			{
//#pragma omp for nowait
//				for (int h = 0; h < height; h++)
//					for (int w = 0; w < width; w++)
//					{
//						T *ptr_dst = matrix + ((b * height + h) * width + w) * shape.w * kernel_size.x * kernel_size.y; // start of a row
//						for (int i = pad_h; i >= -pad_h; i--)
//							for (int j = pad_w; j >= -pad_w; j--, ptr_dst += shape.w)
//								if ((h + i) >= 0 && (h + i) < height && (w + j) >= 0 && (w + j) < width)
//									std::memcpy(ptr_dst, tensor + ((b * height + h + i) * width + w + j) * shape.w, block_length);
//								else
//									std::memset(ptr_dst, 0, block_length);
//					}
//			}
//		}
//	}
}

namespace avocado
{
	namespace backend
	{
//		avStatus_t refIm2Col(avContext_t context, const avConvolution_t config, const avTensor_t input, avTensor_t output)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//
//		avStatus_t refConvolutionBiasActivationForward(avContext_t context, const avConvolution_t config, const avScalar_t alpha1,
//				const avScalar_t beta, const avTensor_t input, avTensor_t output, const avTensor_t weights, const avTensor_t bias,
//				const avActivationType_t activation, const avScalar_t alpha2, const avTensor_t add)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//		avStatus_t refConvolutionForward(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
//				const avTensor_t input, avTensor_t output, const avTensor_t weights)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//		avStatus_t refConvolutionBackward(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
//				avTensor_t gradientPrev, avTensor_t gradientNext, const avTensor_t output, const avTensor_t weights, const avActivationType_t activation)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
//		avStatus_t refConvolutionUpdate(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
//				const avTensor_t input, const avTensor_t gradientNext, avTensor_t weightUpdate, avTensor_t biasUpdate)
//		{
//			return AVOCADO_STATUS_NOT_SUPPORTED;
//		}
	} /* namespace backend */
} /* namespace avocado */

