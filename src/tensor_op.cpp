/*
 * tensor_op.cpp
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

	template<typename T>
	void kernel_concat_tensors(T *dst, const T *src, avSize_t first_dim, avSize_t src_last_dim, avSize_t dst_last_dim, avSize_t dst_offset) noexcept
	{
		for (avSize_t i = 0; i < first_dim; i++)
			for (avSize_t j = 0; j < src_last_dim; j++)
				dst[i * dst_last_dim + dst_offset + j] = src[i * src_last_dim + j];
	}
	template<typename T>
	void concat_helper(const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc[],
			const avMemoryDescriptor_t aMem[], int nbTensors)
	{
		const avSize_t first_dim = getTensor(cDesc).volumeWithoutLastDim();
		const avSize_t dst_last_dim = getTensor(cDesc).lastDim();

		avSize_t last_dim_offset = 0;
		for (int i = 0; i < nbTensors; i++)
		{
			const avSize_t src_last_dim = getTensor(aDesc[i]).lastDim();
			kernel_concat_tensors(getPointer<T>(cMem), getPointer<T>(aMem[i]), first_dim, src_last_dim, dst_last_dim, last_dim_offset);
			last_dim_offset += src_last_dim;
		}
	}

	template<typename T>
	void kernel_split_tensors(T *dst, const T *src, avSize_t first_dim, avSize_t src_last_dim, avSize_t dst_last_dim, avSize_t src_offset) noexcept
	{
		for (avSize_t i = 0; i < first_dim; i++)
			for (avSize_t j = 0; j < dst_last_dim; j++)
				dst[i * dst_last_dim + j] = src[i * src_last_dim + src_offset + j];
	}
	template<typename T>
	void split_helper(const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[], const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, int nbTensors)
	{
		const avSize_t first_dim = getTensor(aDesc).volumeWithoutLastDim();
		const avSize_t src_last_dim = getTensor(aDesc).lastDim();

		avSize_t last_dim_offset = 0;
		for (int i = 0; i < nbTensors; i++)
		{
			const avSize_t dst_last_dim = getTensor(cDesc[i]).lastDim();
			kernel_split_tensors(getPointer<T>(cMem[i]), getPointer<T>(aMem), first_dim, src_last_dim, dst_last_dim, last_dim_offset);
			last_dim_offset += dst_last_dim;
		}
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
	void kernel_add_scalar_to_tensor(T *dst, const void *scalar, avSize_t elements) noexcept
	{
		const T value = getScalarValue<T>(scalar);
		for (avSize_t i = 0; i < elements; i++)
			dst[i] += value;
	}

	template<typename T, typename U, typename V>
	void kernel_add_bias(T *dst, U alpha3, U alpha1, const V *src1, U alpha2, const U *src2, U beta, BroadcastedDimensions dims,
			avActivationType_t type) noexcept
	{
		if (beta == zero<U>())
			clear(dst, volume(dims));

		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
			{
				U lhs = alpha1 * static_cast<U>(src1[i * dims.last + j]);
				U rhs = alpha2 * static_cast<U>(src2[j]);
				U tmp = activation_forward(type, lhs + rhs);
				dst[i * dims.last + j] = Store<T, U>::store(alpha3 * tmp + beta * static_cast<U>(dst[i * dims.last + j]));
			}
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors)
		{
			switch (dataTypeSize(getTensor(cDesc).dtype()))
			{
				case 1:
					concat_helper<int8_t>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				case 2:
					concat_helper<int16_t>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				case 4:
					concat_helper<int32_t>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				case 8:
					concat_helper<int64_t>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				case 16:
					concat_helper<int4>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors)
		{
			switch (dataTypeSize(getTensor(aDesc).dtype()))
			{
				case 1:
					split_helper<int8_t>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				case 2:
					split_helper<int16_t>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				case 4:
					split_helper<int32_t>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				case 8:
					split_helper<int64_t>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				case 16:
					split_helper<int4>(cDesc, cMem, aDesc, aMem, nbTensors);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const int newDimOrder[])
		{
			switch (dataTypeSize(getTensor(aDesc).dtype()))
			{
				case 1:
					kernel_transpose<int8_t>(getPointer<int8_t>(cMem), getPointer<int8_t>(aMem), getTensor(aDesc), newDimOrder);
					break;
				case 2:
					kernel_transpose<int16_t>(getPointer<int16_t>(cMem), getPointer<int16_t>(aMem), getTensor(aDesc), newDimOrder);
					break;
				case 4:
					kernel_transpose<int32_t>(getPointer<int32_t>(cMem), getPointer<int32_t>(aMem), getTensor(aDesc), newDimOrder);
					break;
				case 8:
					kernel_transpose<int64_t>(getPointer<int64_t>(cMem), getPointer<int64_t>(aMem), getTensor(aDesc), newDimOrder);
					break;
				case 16:
					kernel_transpose<int4>(getPointer<int4>(cMem), getPointer<int4>(aMem), getTensor(aDesc), newDimOrder);
					break;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *alpha)
		{
			const avSize_t elements = getTensor(cDesc).volume();
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					kernel_scale_tensor(getPointer<uint8_t>(cMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT8:
					kernel_scale_tensor(getPointer<int8_t>(cMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT16:
					kernel_scale_tensor(getPointer<int16_t>(cMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT32:
					kernel_scale_tensor(getPointer<int32_t>(cMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT64:
					kernel_scale_tensor(getPointer<int64_t>(cMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_scale_tensor(getPointer<float16>(cMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_scale_tensor(getPointer<bfloat16>(cMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_scale_tensor(getPointer<float>(cMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_scale_tensor(getPointer<double>(cMem), getAlphaValue<double>(alpha), elements);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					kernel_scale_tensor(getPointer<std::complex<float>>(cMem), getAlphaValue<std::complex<float>>(alpha), elements);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					kernel_scale_tensor(getPointer<std::complex<double>>(cMem), getAlphaValue<std::complex<double>>(alpha), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const void *scalar)
		{
			const avSize_t elements = getTensor(cDesc).volume();
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					kernel_add_scalar_to_tensor(getPointer<uint8_t>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_INT8:
					kernel_add_scalar_to_tensor(getPointer<int8_t>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_INT16:
					kernel_add_scalar_to_tensor(getPointer<int16_t>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_INT32:
					kernel_add_scalar_to_tensor(getPointer<int32_t>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_INT64:
					kernel_add_scalar_to_tensor(getPointer<int64_t>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_scalar_to_tensor(getPointer<float16>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_add_scalar_to_tensor(getPointer<bfloat16>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_scalar_to_tensor(getPointer<float>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_scalar_to_tensor(getPointer<double>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					kernel_add_scalar_to_tensor(getPointer<std::complex<float>>(cMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					kernel_add_scalar_to_tensor(getPointer<std::complex<double>>(cMem), scalar, elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refAddBias(avContextDescriptor_t context, const void *alpha3, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, avActivationType_t activation)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(cDesc), getTensor(bDesc));
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
				{
					switch (getTensor(aDesc).dtype())
					{
						case AVOCADO_DTYPE_INT8:
							kernel_add_bias(getPointer<int8_t>(cMem), getAlphaValue(alpha3), getAlphaValue(alpha1), getPointer<int8_t>(aMem),
									getAlphaValue(alpha2), getPointer<float>(bMem), getBetaValue(beta), dimensions, activation);
							break;
						case AVOCADO_DTYPE_INT32:
							kernel_add_bias(getPointer<int8_t>(cMem), getAlphaValue(alpha3), getAlphaValue(alpha1), getPointer<int32_t>(aMem),
									getAlphaValue(alpha2), getPointer<float>(bMem), getBetaValue(beta), dimensions, activation);
							break;
						default:
							return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
					}
					break;
				}
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_bias(getPointer<float16>(cMem), getAlphaValue(alpha3), getAlphaValue(alpha1), getPointer<float16>(aMem),
							getAlphaValue(alpha2), getPointer<float>(bMem), getBetaValue(beta), dimensions, activation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_add_bias(getPointer<bfloat16>(cMem), getAlphaValue(alpha3), getAlphaValue(alpha1), getPointer<bfloat16>(aMem),
							getAlphaValue(alpha2), getPointer<float>(bMem), getBetaValue(beta), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_bias(getPointer<float>(cMem), getAlphaValue(alpha3), getAlphaValue(alpha1), getPointer<float>(aMem),
							getAlphaValue(alpha2), getPointer<float>(bMem), getBetaValue(beta), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_bias(getPointer<double>(cMem), getAlphaValue<double>(alpha3), getAlphaValue<double>(alpha1), getPointer<double>(aMem),
							getAlphaValue<double>(alpha2), getPointer<double>(bMem), getBetaValue<double>(beta), dimensions, activation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

