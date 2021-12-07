/*
 * basic_math.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */
#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

#include "fp16.hpp"
#include "utils.hpp"
#include "activations.hpp"

#include <memory>

namespace
{
	using namespace avocado::backend;

	struct int4
	{
			int x, y, z, w;
	};

	template<typename T, typename U>
	struct Converter
	{
			static T convert(U x) noexcept
			{
				return static_cast<T>(x);
			}
	};

	template<typename U>
	struct Converter<float16, U>
	{
			static float16 convert(U x) noexcept
			{
				return float16(static_cast<float>(x));
			}
	};
	template<typename T>
	struct Converter<T, float16>
	{
			static T convert(float16 x) noexcept
			{
				return static_cast<T>(static_cast<float>(x));
			}
	};
	template<>
	struct Converter<float16, float16>
	{
			static float16 convert(float16 x) noexcept
			{
				return x;
			}
	};
	template<>
	struct Converter<float16, std::complex<float>>
	{
			static float16 convert(std::complex<float> x) noexcept
			{
				return Converter<float16, float>::convert(x.real());
			}
	};
	template<>
	struct Converter<float16, std::complex<double>>
	{
			static float16 convert(std::complex<double> x) noexcept
			{
				return Converter<float16, double>::convert(x.real());
			}
	};

	template<typename U>
	struct Converter<bfloat16, U>
	{
			static bfloat16 convert(U x) noexcept
			{
				return bfloat16(static_cast<float>(x));
			}
	};
	template<typename T>
	struct Converter<T, bfloat16>
	{
			static T convert(bfloat16 x) noexcept
			{
				return static_cast<T>(static_cast<float>(x));
			}
	};
	template<>
	struct Converter<bfloat16, bfloat16>
	{
			static bfloat16 convert(bfloat16 x) noexcept
			{
				return x;
			}
	};
	template<>
	struct Converter<bfloat16, std::complex<float>>
	{
			static bfloat16 convert(std::complex<float> x) noexcept
			{
				return Converter<bfloat16, float>::convert(x.real());
			}
	};
	template<>
	struct Converter<bfloat16, std::complex<double>>
	{
			static bfloat16 convert(std::complex<double> x) noexcept
			{
				return Converter<bfloat16, double>::convert(x.real());
			}
	};

	template<>
	struct Converter<float16, bfloat16>
	{
			static float16 convert(bfloat16 x) noexcept
			{
				return float16(static_cast<float>(x));
			}
	};
	template<>
	struct Converter<bfloat16, float16>
	{
			static bfloat16 convert(float16 x) noexcept
			{
				return bfloat16(static_cast<float>(x));
			}
	};

	template<typename T>
	struct Converter<T, std::complex<float>>
	{
			static T convert(std::complex<float> x) noexcept
			{
				return Converter<T, float>::convert(x.real());
			}
	};
	template<typename T>
	struct Converter<T, std::complex<double>>
	{
			static T convert(std::complex<double> x) noexcept
			{
				return Converter<T, double>::convert(x.real());
			}
	};

	template<typename T, typename U>
	void kernel_convert(T *dst, const U *src, avSize_t elements) noexcept
	{
		for (avSize_t i = 0; i < elements; i++)
			dst[i] = Converter<T, U>::convert(src[i]);
	}
	template<typename T>
	void convert_helper(T *dst, const void *src, avSize_t elements, avDataType_t srcType)
	{
		assert(dst != nullptr);
		assert(src != nullptr);

		switch (srcType)
		{
			case AVOCADO_DTYPE_UINT8:
				kernel_convert(dst, reinterpret_cast<const uint8_t>(src), elements);
				break;
			case AVOCADO_DTYPE_INT8:
				kernel_convert(dst, reinterpret_cast<const int8_t>(src), elements);
				break;
			case AVOCADO_DTYPE_INT16:
				kernel_convert(dst, reinterpret_cast<const int16_t>(src), elements);
				break;
			case AVOCADO_DTYPE_INT32:
				kernel_convert(dst, reinterpret_cast<const int32_t>(src), elements);
				break;
			case AVOCADO_DTYPE_INT64:
				kernel_convert(dst, reinterpret_cast<const int64_t>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT16:
				kernel_convert(dst, reinterpret_cast<const float16>(src), elements);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_convert(dst, reinterpret_cast<const bfloat16>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_convert(dst, reinterpret_cast<const float>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_convert(dst, reinterpret_cast<const double>(src), elements);
				break;
			case AVOCADO_DTYPE_COMPLEX32:
				kernel_convert(dst, reinterpret_cast<const std::complex<float>>(src), elements);
				break;
			case AVOCADO_DTYPE_COMPLEX64:
				kernel_convert(dst, reinterpret_cast<const std::complex<double>>(src), elements);
				break;
			default:
				break;
		}
	}

	template<typename T>
	void kernel_concat_tensors(T *dst, const T *src, avSize_t first_dim, avSize_t src_last_dim, avSize_t dst_last_dim, avSize_t dst_offset) noexcept
	{
		for (avSize_t i = 0; i < first_dim; i++)
			for (avSize_t j = 0; j < src_last_dim; j++)
				dst[i * dst_last_dim + dst_offset + j] = src[i * src_last_dim + j];
	}
	template<typename T>
	void kernel_split_tensors(T *dst, const T *src, avSize_t first_dim, avSize_t src_last_dim, avSize_t dst_last_dim, avSize_t src_offset) noexcept
	{
		for (avSize_t i = 0; i < first_dim; i++)
			for (avSize_t j = 0; j < dst_last_dim; j++)
				dst[i * dst_last_dim + j] = src[i * src_last_dim + src_offset + j];
	}

	template<typename T>
	void kernel_transpose(T *dst, const T *src, const TensorDescriptor &src_shape, const int *ordering)
	{
		const int src_volume = src_shape.volume();
		const int dimension = src_shape.nbDims();

		std::unique_ptr<int[]> src_stride = std::make_unique<int[]>(dimension);
		std::unique_ptr<int[]> dst_stride = std::make_unique<int[]>(dimension);

		int tmp_src = 1, tmp_dst = 1;
		for (int i = dimension - 1; i >= 0; i--)
		{
			src_stride[i] = tmp_src;
			dst_stride[ordering[i]] = tmp_dst;
			tmp_src *= src_shape.dimension(i);
			tmp_dst *= src_shape.dimension(ordering[i]);
		}

		for (int i = 0; i < src_volume; i++)
		{
			int src_idx = i, dst_idx = 0;
			for (int j = 0; j < dimension; j++)
			{
				int tmp = src_idx / src_stride[j];
				dst_idx += tmp * dst_stride[j];
				src_idx -= tmp * src_stride[j];
			}
			dst[dst_idx] = src[i];
		}
	}

	template<typename T, typename U>
	void kernel_scale_tensor(T *dst, U value, avSize_t elements) noexcept
	{
		for (avSize_t i = 0; i < elements; i++)
			dst[i] = static_cast<U>(dst[i]) * value;
	}
	template<typename T>
	void kernel_add_scalar_to_tensor(T *dst, T value, avSize_t elements) noexcept
	{
		for (avSize_t i = 0; i < elements; i++)
			dst[i] += value;
	}
	template<typename T, typename U>
	void kernel_op_tensor(T *dst, const T *src1, const T *src2, U alpha1, U alpha2, U beta, BroadcastedDimensions dims, avOpTensorOp_t operation)
	noexcept
	{
		if (beta == zero<U>())
			clear(dst, volume(dims));

		for (avSize_t i = 0; i < dims.first; i++)
		{
			for (avSize_t j = 0; j < dims.last; j++)
			{
				T value1 = static_cast<T>(alpha1 * static_cast<U>(src1[i * dims.last + j]));
				T value2 = static_cast<T>(alpha2 * static_cast<U>(src2[i * dims.last + j]));
				T result = zero<T>();
				switch (operation)
				{
					case AVOCADO_OP_TENSOR_ADD:
						result = value1 + value2;
						break;
					case AVOCADO_OP_TENSOR_SUB:
						result = value1 - value2;
						break;
					case AVOCADO_OP_TENSOR_MUL:
						result = value1 * value2;
						break;
					case AVOCADO_OP_TENSOR_MIN:
						result = std::min(value1, value2);
						break;
					case AVOCADO_OP_TENSOR_MAX:
						result = std::max(value1, value2);
						break;
				}
				dst[i * dims.last + j] = static_cast<U>(result) + beta * static_cast<U>(dst[i * dims.last + j]);
			}
		}
	}
	template<typename T, typename U>
	void kernel_op_single_tensor(T *dst, const T *src, U alpha, U beta, avSize_t elements, avOpSingleTensorOp_t operation)
	noexcept
	{
		if (beta == zero<U>())
			clear(dst, elements);

		for (avSize_t i = 0; i < elements; i++)
		{
			T value = static_cast<T>(alpha * static_cast<U>(src[i]));
			T result = zero<T>();
			switch (operation)
			{
				case AVOCADO_OP_SINGLE_TENSOR_ABS:
					result = avocado::backend::fabs(value);
					break;
				case AVOCADO_OP_SINGLE_TENSOR_SQUARE:
					result = square(value);
					break;
				case AVOCADO_OP_SINGLE_TENSOR_SQRT:
					result = sqrt(value);
					break;
				case AVOCADO_OP_SINGLE_TENSOR_NOT:
					result = -value;
					break;
			}
			dst[i] = static_cast<U>(result) + beta * static_cast<U>(dst[i]);
		}
	}
	template<typename T, typename U>
	void kernel_reduce_tensor(T *dst, const T *src, U alpha, U beta, BroadcastedDimensions dims, avReduceTensorOp_t operation) noexcept
	{
		std::unique_ptr<T[]> workspace = std::make_unique<T[]>(dims.last);

		// first perform initialization of workspace tensor
		for (avSize_t j = 0; j < dims.last; j++)
		{
			switch (operation)
			{
				case AVOCADO_REDUCE_TENSOR_ADD:
				case AVOCADO_REDUCE_TENSOR_AVG:
				case AVOCADO_REDUCE_TENSOR_NORM1:
				case AVOCADO_REDUCE_TENSOR_NORM2:
					workspace[j] = zero<T>();
					break;
				case AVOCADO_REDUCE_TENSOR_MUL:
				case AVOCADO_REDUCE_TENSOR_MUL_NO_ZEROS:
					workspace[j] = one<T>();
					break;
				case AVOCADO_REDUCE_TENSOR_MIN:
				case AVOCADO_REDUCE_TENSOR_MAX:
					workspace[j] = src[j];
					break;
				case AVOCADO_REDUCE_TENSOR_AMAX:
					workspace[j] = avocado::backend::fabs(src[j]);
					break;
			}
		}

		// now perform the main reduction
		for (avSize_t i = 0; i < dims.first; i++)
		{
			for (avSize_t j = 0; j < dims.last; j++)
			{
				T value = src[i * dims.last + j];
				switch (operation)
				{
					case AVOCADO_REDUCE_TENSOR_ADD:
						workspace[j] += value;
						break;
					case AVOCADO_REDUCE_TENSOR_MUL:
						workspace[j] *= value;
						break;
					case AVOCADO_REDUCE_TENSOR_MIN:
						workspace[j] = std::min(workspace[j], value);
						break;
					case AVOCADO_REDUCE_TENSOR_MAX:
						workspace[j] = std::max(workspace[j], value);
						break;
					case AVOCADO_REDUCE_TENSOR_AMAX:
						workspace[j] = std::max(workspace[j], avocado::backend::fabs(value));
						break;
					case AVOCADO_REDUCE_TENSOR_AVG:
						workspace[j] += value;
						break;
					case AVOCADO_REDUCE_TENSOR_NORM1:
						workspace[j] += avocado::backend::fabs(value);
						break;
					case AVOCADO_REDUCE_TENSOR_NORM2:
						workspace[j] += square(value);
						break;
					case AVOCADO_REDUCE_TENSOR_MUL_NO_ZEROS:
						if (value != zero<T>())
							workspace[j] *= value;
						break;
				}
			}
		}

		// now perform final action, if necessary
		switch (operation)
		{
			default:
				break;
			case AVOCADO_REDUCE_TENSOR_AVG:
				for (avSize_t j = 0; j < dims.last; j++)
					workspace[j] = static_cast<U>(workspace[j]) / dims.first;
				break;
			case AVOCADO_REDUCE_TENSOR_NORM2:
				for (avSize_t j = 0; j < dims.last; j++)
					workspace[j] = sqrt(static_cast<U>(workspace[j]));
				break;
		}

		if (beta == zero<U>())
			clear(dst, dims.last);
		// now store result in dst
		for (avSize_t j = 0; j < dims.last; j++)
			dst[j] = alpha * static_cast<U>(workspace[j]) + beta * static_cast<U>(dst[j]);
	}
	template<typename T, typename U>
	void kernel_add_tensors(T *dst, const T *src, U alpha, U beta, BroadcastedDimensions dims, avActivationType_t type) noexcept
	{
		if (beta == zero<U>())
			clear(dst, volume(dims));

		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
				dst[i * dims.last + j] = activation_forward(type, alpha * static_cast<U>(src[j]) + beta * static_cast<U>(dst[i * dims.last + j]));
	}
}

namespace avocado
{
	namespace backend
	{
//		avStatus_t refChangeType(avContext_t context, void *dst, avDataType_t dstType, const void *src, avDataType_t srcType, avSize_t elements)
//		{
//			/* context can be null */
//			assert(dst != nullptr);
//			assert(src != nullptr);
//			switch (dstType)
//			{
//				case AVOCADO_DTYPE_UINT8:
//					convert_helper(data < uint8_t > (dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_INT8:
//					convert_helper(data < int8_t > (dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_INT16:
//					convert_helper(data < int16_t > (dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_INT32:
//					convert_helper(data < int32_t > (dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_INT64:
//					convert_helper(data < int64_t > (dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_FLOAT16:
//					convert_helper(data < float16 > (dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					convert_helper(data < bfloat16 > (dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					convert_helper(data<float>(dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					convert_helper(data<double>(dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					convert_helper(data<std::complex<float>>(dst), src, elements, srcType);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					convert_helper(data<std::complex<double>>(dst), src, elements, srcType);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//
//		avStatus_t refConcatTensors(avContext_t context, avTensor_t dst, const avTensor_t src, avSize_t lastDimOffsetInBytes)
//		{
//			assert(same_type(dst, src));
//			const int type_size = dataTypeSize(dst->dtype);
//			const avSize_t first_dim = volumeWithoutLastDim(dst) * type_size;
//			const avSize_t src_last_dim = lastDim(src) * type_size;
//			const avSize_t dst_last_dim = lastDim(dst) * type_size;
//
//			switch (type_size)
//			{
//				case 1:
//					kernel_concat_tensors(data < int8_t > (dst), data < int8_t > (src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
//					break;
//				case 2:
//					kernel_concat_tensors(data < int16_t > (dst), data < int16_t > (src), first_dim, src_last_dim, dst_last_dim,
//							lastDimOffsetInBytes);
//					break;
//				case 4:
//					kernel_concat_tensors(data < int32_t > (dst), data < int32_t > (src), first_dim, src_last_dim, dst_last_dim,
//							lastDimOffsetInBytes);
//					break;
//				case 8:
//					kernel_concat_tensors(data < int64_t > (dst), data < int64_t > (src), first_dim, src_last_dim, dst_last_dim,
//							lastDimOffsetInBytes);
//					break;
//				case 16:
//					kernel_concat_tensors(data < int4 > (dst), data < int4 > (src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
//					break;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refSplitTensors(avContext_t context, avTensor_t dst, const avTensor_t src, avSize_t lastDimOffsetInBytes)
//		{
//			assert(same_type(dst, src));
//			const int type_size = dataTypeSize(dst->dtype);
//			const avSize_t first_dim = volumeWithoutLastDim(dst) * type_size;
//			const avSize_t src_last_dim = lastDim(src) * type_size;
//			const avSize_t dst_last_dim = lastDim(dst) * type_size;
//
//			switch (type_size)
//			{
//				case 1:
//					kernel_split_tensors(data < int8_t > (dst), data < int8_t > (src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
//					break;
//				case 2:
//					kernel_split_tensors(data < int16_t > (dst), data < int16_t > (src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
//					break;
//				case 4:
//					kernel_split_tensors(data < int32_t > (dst), data < int32_t > (src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
//					break;
//				case 8:
//					kernel_split_tensors(data < int64_t > (dst), data < int64_t > (src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
//					break;
//				case 16:
//					kernel_split_tensors(data < int4 > (dst), data < int4 > (src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
//					break;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refTranspose(avContext_t context, avTensor_t dst, const avTensor_t src, const int *order)
//		{
//			assert(context != nullptr);
//			assert(dst != nullptr);
//			assert(src != nullptr);
//			assert(order != nullptr);
//			switch (dataTypeSize(dst->dtype))
//			{
//				case 1:
//					kernel_transpose<int8_t>(data < int8_t > (dst), data < int8_t > (src), src->shape, order);
//					break;
//				case 2:
//					kernel_transpose<int16_t>(data < int16_t > (dst), data < int16_t > (src), src->shape, order);
//					break;
//				case 4:
//					kernel_transpose<int32_t>(data < int32_t > (dst), data < int32_t > (src), src->shape, order);
//					break;
//				case 8:
//					kernel_transpose<int64_t>(data < int64_t > (dst), data < int64_t > (src), src->shape, order);
//					break;
//				case 16:
//					kernel_transpose<int4>(data < int4 > (dst), data < int4 > (src), src->shape, order);
//					break;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//
//		avStatus_t refScaleTensor(avContext_t context, avTensor_t dst, const avScalar_t src)
//		{
//			assert(dst != nullptr);
//			assert(src != nullptr);
//			assert(same_type(dst, src));
//
//			const avSize_t elements = volume(dst);
//			switch (dst->dtype)
//			{
//				case AVOCADO_DTYPE_UINT8:
//					kernel_scale_tensor(data < uint8_t > (dst), getAlphaValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_INT8:
//					kernel_scale_tensor(data < int8_t > (dst), getAlphaValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_INT16:
//					kernel_scale_tensor(data < int16_t > (dst), getAlphaValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_INT32:
//					kernel_scale_tensor(data < int32_t > (dst), getAlphaValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_INT64:
//					kernel_scale_tensor(data < int64_t > (dst), getAlphaValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_scale_tensor(data < float16 > (dst), getAlphaValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_scale_tensor(data < bfloat16 > (dst), getAlphaValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_scale_tensor(data<float>(dst), getAlphaValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_scale_tensor(data<double>(dst), getAlphaValue<double>(src), elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_scale_tensor(data<std::complex<float>>(dst), getAlphaValue<std::complex<float>>(src), elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_scale_tensor(data<std::complex<double>>(dst), getAlphaValue<std::complex<double>>(src), elements);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refAddScalarToTensor(avContext_t context, avTensor_t dst, const avScalar_t src)
//		{
//			assert(dst != nullptr);
//			assert(src != nullptr);
//			assert(same_type(dst, src));
//
//			const avSize_t elements = volume(dst);
//			switch (dst->dtype)
//			{
//				case AVOCADO_DTYPE_UINT8:
//					kernel_add_scalar_to_tensor(data < uint8_t > (dst), getScalarValue < uint8_t > (src), elements);
//					break;
//				case AVOCADO_DTYPE_INT8:
//					kernel_add_scalar_to_tensor(data < int8_t > (dst), getScalarValue < int8_t > (src), elements);
//					break;
//				case AVOCADO_DTYPE_INT16:
//					kernel_add_scalar_to_tensor(data < int16_t > (dst), getScalarValue < int16_t > (src), elements);
//					break;
//				case AVOCADO_DTYPE_INT32:
//					kernel_add_scalar_to_tensor(data < int32_t > (dst), getScalarValue < int32_t > (src), elements);
//					break;
//				case AVOCADO_DTYPE_INT64:
//					kernel_add_scalar_to_tensor(data < int64_t > (dst), getScalarValue < int64_t > (src), elements);
//					break;
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_add_scalar_to_tensor(data < float16 > (dst), getScalarValue < float16 > (src), elements);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_add_scalar_to_tensor(data < bfloat16 > (dst), getScalarValue < bfloat16 > (src), elements);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_add_scalar_to_tensor(data<float>(dst), getScalarValue<float>(src), elements);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_add_scalar_to_tensor(data<double>(dst), getScalarValue<double>(src), elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_add_scalar_to_tensor(data<std::complex<float>>(dst), getScalarValue<std::complex<float>>(src), elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_add_scalar_to_tensor(data<std::complex<double>>(dst), getScalarValue<std::complex<double>>(src), elements);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//
//		avStatus_t refOpTensor(avContext_t context, avOpTensorOp_t operation, const avScalar_t alpha1, const avTensor_t input1,
//				const avScalar_t alpha2, const avTensor_t input2, const avScalar_t beta, avTensor_t output)
//		{
//			assert(context != nullptr);
//			assert(output != nullptr);
//			assert(input1 != nullptr);
//			assert(input2 != nullptr);
//			assert(same_shape(output, input1));
//			assert(same_type(output, input1, input2));
//
//			BroadcastedDimensions dimensions = getBroadcastDimensions(input1, input2);
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_op_tensor(data < float16 > (output), data < float16 > (input1), data < float16 > (input2), getAlphaValue(alpha1),
//							getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_op_tensor(data < bfloat16 > (output), data < bfloat16 > (input1), data < bfloat16 > (input2), getAlphaValue(alpha1),
//							getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_op_tensor(data<float>(output), data<float>(input1), data<float>(input2), getAlphaValue(alpha1), getAlphaValue(alpha2),
//							getBetaValue(beta), dimensions, operation);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_op_tensor(data<double>(output), data<double>(input1), data<double>(input2), getAlphaValue<double>(alpha1),
//							getAlphaValue<double>(alpha2), getBetaValue<double>(beta), dimensions, operation);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t cudaOpSingleTensor(avContext_t context, avOpSingleTensorOp_t operation, const avScalar_t alpha, const avTensor_t input,
//				const avScalar_t beta, avTensor_t output)
//		{
//			assert(context != nullptr);
//			assert(output != nullptr);
//			assert(input != nullptr);
//			assert(same_shape(output, input));
//			assert(same_type(output, input));
//
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_op_single_tensor(data < float16 > (output), data < float16 > (input), getAlphaValue(alpha), getBetaValue(beta),
//							volume(input), operation);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_op_single_tensor(data < bfloat16 > (output), data < bfloat16 > (input), getAlphaValue(alpha), getBetaValue(beta),
//							volume(input), operation);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_op_single_tensor(data<float>(output), data<float>(input), getAlphaValue(alpha), getBetaValue(beta), volume(input),
//							operation);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_op_single_tensor(data<double>(output), data<double>(input), getAlphaValue<double>(alpha), getBetaValue<double>(beta),
//							volume(input), operation);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refReduceTensor(avContext_t context, avReduceTensorOp_t operation, const avScalar_t alpha, const avScalar_t beta,
//				const avTensor_t input, avTensor_t output)
//		{
//			assert(context != nullptr);
//			assert(output != nullptr);
//			assert(input != nullptr);
//			assert(same_type(output, input));
//
//			BroadcastedDimensions dimensions = getBroadcastDimensions(input, output);
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_reduce_tensor(data < float16 > (output), data < float16 > (input), getAlphaValue(alpha), getBetaValue(beta), dimensions,
//							operation);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_reduce_tensor(data < bfloat16 > (output), data < bfloat16 > (input), getAlphaValue(alpha), getBetaValue(beta), dimensions,
//							operation);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_reduce_tensor(data<float>(output), data<float>(input), getAlphaValue(alpha), getBetaValue(beta), dimensions, operation);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_reduce_tensor(data<double>(output), data<double>(input), getAlphaValue<double>(alpha), getBetaValue<double>(beta),
//							dimensions, operation);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//
//		avStatus_t refAddTensors(avContext_t context, const avScalar_t alpha, const avScalar_t beta, avTensor_t output, const avTensor_t bias,
//				avActivationType_t activation)
//		{
//			assert(context != nullptr);
//			assert(output != nullptr);
//			assert(bias != nullptr);
//			assert(same_type(output, bias));
//
//			BroadcastedDimensions dimensions = getBroadcastDimensions(output, bias);
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_add_tensors(data < float16 > (output), data < float16 > (bias), getAlphaValue(alpha), getBetaValue(beta), dimensions,
//							activation);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_add_tensors(data < bfloat16 > (output), data < bfloat16 > (bias), getAlphaValue(alpha), getBetaValue(beta), dimensions,
//							activation);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_add_tensors(data<float>(output), data<float>(bias), getAlphaValue(alpha), getBetaValue(beta), dimensions, activation);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_add_tensors(data<double>(output), data<double>(bias), getAlphaValue<double>(alpha), getBetaValue<double>(beta), dimensions,
//							activation);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}

	} /* namespace backend */
} /* namespace avocado */

