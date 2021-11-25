/*
 * basic_math.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */
#include <avocado/reference_backend.h>
#include <avocado/backend/tensor_helpers.hpp>

#include "utils.hpp"

#include <memory>

namespace
{
	using namespace avocado::backend;

	struct int4
	{
			int x, y, z, w;
	};

	template<typename T>
	void kernel_concat_tensors(T *dst, const T *src, size_t first_dim, size_t src_last_dim, size_t dst_last_dim, size_t dst_offset) noexcept
	{
		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < src_last_dim; j++)
				dst[i * dst_last_dim + dst_offset + j] = src[i * src_last_dim + j];
	}
	template<typename T>
	void kernel_split_tensors(T *dst, const T *src, size_t first_dim, size_t src_last_dim, size_t dst_last_dim, size_t src_offset) noexcept
	{
		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < dst_last_dim; j++)
				dst[i * dst_last_dim + j] = src[i * src_last_dim + src_offset + j];
	}
	template<typename T>
	void kernel_transpose(T *dst, const T *src, const ShapeDescriptor &src_shape, const int *ordering, int dimension)
	{
		const int src_volume = volume(src_shape);

		std::unique_ptr<int[]> src_stride = std::make_unique<int[]>(dimension);
		std::unique_ptr<int[]> dst_stride = std::make_unique<int[]>(dimension);

		int tmp_src = 1, tmp_dst = 1;
		for (int i = dimension - 1; i >= 0; i--)
		{
			src_stride[i] = tmp_src;
			dst_stride[ordering[i]] = tmp_dst;
			tmp_src *= src_shape.dim[i];
			tmp_dst *= src_shape.dim[ordering[i]];
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
	template<typename T>
	void kernel_scale_tensor(T *dst, T value, size_t elements) noexcept
	{
		for (size_t i = 0; i < elements; i++)
			dst[i] *= value;
	}
	template<typename T>
	void kernel_add_scalar_to_tensor(T *dst, T value, size_t elements) noexcept
	{
		for (size_t i = 0; i < elements; i++)
			dst[i] += value;
	}
	template<typename T>
	void kernel_op_tensor(T *dst, const T *src1, const T *src2, T alpha1, T alpha2, T beta, size_t first_dim, size_t last_dim,
			avOpTensorOp_t operation) noexcept
	{
		if (beta == zero<T>())
			clear(dst, first_dim * last_dim);

		for (size_t i = 0; i < first_dim; i++)
		{
			for (size_t j = 0; j < last_dim; j++)
			{
				T value1 = alpha1 * src1[last_dim];
				T value2 = alpha2 * src2[last_dim];
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
					case AVOCADO_OP_TENSOR_SQRT:
						result = sqrt(value1);
						break;
				}
				dst[i * last_dim + j] = result + beta * dst[i * last_dim + j];
			}
		}
	}
	template<typename T>
	void kernel_reduce_tensor(T *dst, const T *src, T alpha, T beta, size_t first_dim, size_t last_dim, avReduceTensorOp_t operation) noexcept
	{
		std::unique_ptr<T[]> workspace = std::make_unique<T[]>(last_dim);
		clear(workspace.get(), last_dim);

		for (size_t i = 0; i < first_dim; i++)
		{
			for (size_t j = 0; j < last_dim; j++)
			{
				T value = src[i * last_dim + j];
				T result = zero<T>();
				switch (operation)
				{
					case AVOCADO_REDUCE_TENSOR_ADD:
						result += value;
						break;
					case AVOCADO_REDUCE_TENSOR_MUL:
						result *= value;
						break;
					case AVOCADO_REDUCE_TENSOR_MIN:
						result = std::min(result, value);
						break;
					case AVOCADO_REDUCE_TENSOR_MAX:
						result = std::max(result, value);
						break;
					case AVOCADO_REDUCE_TENSOR_AMAX:
						result = std::max(abs(result), abs(value));
						break;
					case AVOCADO_REDUCE_TENSOR_AVG:
						result += value;
						break;
					case AVOCADO_REDUCE_TENSOR_NORM1:
						result += abs(value);
						break;
					case AVOCADO_REDUCE_TENSOR_NORM2:
						result += square(value);
						break;
					case AVOCADO_REDUCE_TENSOR_MUL_NO_ZEROS:
						if (value != zero<T>())
							result *= value;
						break;
				}
				workspace[j] += result;
			}
		}
		if (beta == zero<T>())
			clear(dst, last_dim);
		switch (operation)
		{
			default:
				for (size_t j = 0; j < last_dim; j++)
					dst[j] = alpha * workspace[j] + beta * dst[j];
				break;
			case AVOCADO_REDUCE_TENSOR_AVG:
				for (size_t j = 0; j < last_dim; j++)
					dst[j] = alpha * workspace[j] / first_dim + beta * dst[j];
				break;
			case AVOCADO_REDUCE_TENSOR_NORM2:
				for (size_t j = 0; j < last_dim; j++)
					dst[j] = alpha * sqrt(workspace[j]) + beta * dst[j];
				break;
		}
	}
	template<typename T>
	void kernel_add_tensors(T *dst, const T *src, T alpha, T beta, size_t first_dim, size_t last_dim, avActivationType_t type) noexcept
	{
		if (beta == zero<T>())
			clear(dst, first_dim * last_dim);

		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < last_dim; j++)
				dst[i * last_dim + j] = activation_forward(type, alpha * src[j] + beta * dst[i * last_dim + j]);
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refChangeType(avContext_t context, void *dst, avDataType_t dstType, const void *src, avDataType_t srcType, size_t elements)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t refConcatTensors(avContext_t context, avTensor_t dst, const avTensor_t src, size_t lastDimOffsetInBytes)
		{
			assert(same_type(dst, src));
			const int type_size = dataTypeSize(dst->dtype);
			const size_t first_dim = volumeWithoutLastDim(dst) * type_size;
			const size_t src_last_dim = lastDim(src) * type_size;
			const size_t dst_last_dim = lastDim(dst) * type_size;

			switch (type_size)
			{
				case 1:
					kernel_concat_tensors(data<int8_t>(dst), data<int8_t>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
				case 2:
					kernel_concat_tensors(data<int16_t>(dst), data<int16_t>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
				case 4:
					kernel_concat_tensors(data<int32_t>(dst), data<int32_t>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
				case 8:
					kernel_concat_tensors(data<int64_t>(dst), data<int64_t>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
				case 16:
					kernel_concat_tensors(data<int4>(dst), data<int4>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refSplitTensors(avContext_t context, avTensor_t dst, const avTensor_t src, size_t lastDimOffsetInBytes)
		{
			assert(same_type(dst, src));
			const int type_size = dataTypeSize(dst->dtype);
			const size_t first_dim = volumeWithoutLastDim(dst) * type_size;
			const size_t src_last_dim = lastDim(src) * type_size;
			const size_t dst_last_dim = lastDim(dst) * type_size;

			switch (type_size)
			{
				case 1:
					kernel_split_tensors(data<int8_t>(dst), data<int8_t>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
				case 2:
					kernel_split_tensors(data<int16_t>(dst), data<int16_t>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
				case 4:
					kernel_split_tensors(data<int32_t>(dst), data<int32_t>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
				case 8:
					kernel_split_tensors(data<int64_t>(dst), data<int64_t>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
				case 16:
					kernel_split_tensors(data<int4>(dst), data<int4>(src), first_dim, src_last_dim, dst_last_dim, lastDimOffsetInBytes);
					break;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refTranspose(avContext_t context, avTensor_t dst, const avTensor_t src, const int *order)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refScaleTensor(avContext_t context, avTensor_t dst, const avScalar_t src)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refAddScalarToTensor(avContext_t context, avTensor_t dst, const avScalar_t src)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refOpTensor(avContext_t context, avOpTensorOp_t operation, const avScalar_t alpha1, const avTensor_t input1,
				const avScalar_t alpha2, const avTensor_t input2, const avScalar_t beta, avTensor_t output)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refReduceTensor(avContext_t context, avReduceTensorOp_t operation, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refAddTensors(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input, avTensor_t output,
				avActivation_t activation)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

