/*
 * tensor_op.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */
#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

#include "fp16.hpp"
#include "utils.hpp"
#include "activations.hpp"

#include <memory>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::reference;

	struct int4
	{
			int x, y, z, w;
	};

	template<typename T>
	void kernel_concat_tensors(T *dst, const T *src, av_int64 first_dim, av_int64 src_last_dim, av_int64 dst_last_dim, av_int64 dst_offset) noexcept
	{
		for (av_int64 i = 0; i < first_dim; i++)
			for (av_int64 j = 0; j < src_last_dim; j++)
				dst[i * dst_last_dim + dst_offset + j] = src[i * src_last_dim + j];
	}
	template<typename T>
	void concat_helper(const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc[],
			const avMemoryDescriptor_t aMem[], int nbTensors)
	{
		const av_int64 first_dim = getTensor(cDesc).volumeWithoutLastDim();
		const av_int64 dst_last_dim = getTensor(cDesc).lastDim();

		av_int64 last_dim_offset = 0;
		for (int i = 0; i < nbTensors; i++)
		{
			const av_int64 src_last_dim = getTensor(aDesc[i]).lastDim();
			kernel_concat_tensors(getPointer<T>(cMem), getPointer<T>(aMem[i]), first_dim, src_last_dim, dst_last_dim, last_dim_offset);
			last_dim_offset += src_last_dim;
		}
	}

	template<typename T>
	void kernel_split_tensors(T *dst, const T *src, av_int64 first_dim, av_int64 src_last_dim, av_int64 dst_last_dim, av_int64 src_offset) noexcept
	{
		for (av_int64 i = 0; i < first_dim; i++)
			for (av_int64 j = 0; j < dst_last_dim; j++)
				dst[i * dst_last_dim + j] = src[i * src_last_dim + src_offset + j];
	}
	template<typename T>
	void split_helper(const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[], const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, int nbTensors)
	{
		const av_int64 first_dim = getTensor(aDesc).volumeWithoutLastDim();
		const av_int64 src_last_dim = getTensor(aDesc).lastDim();

		av_int64 last_dim_offset = 0;
		for (int i = 0; i < nbTensors; i++)
		{
			const av_int64 dst_last_dim = getTensor(cDesc[i]).lastDim();
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
	void kernel_scale_tensor(T *dst, const T *src, U value, av_int64 elements) noexcept
	{
		for (av_int64 i = 0; i < elements; i++)
			dst[i] = static_cast<U>(src[i]) * value;
	}
	template<typename T>
	void kernel_add_scalar_to_tensor(T *dst, const T *src, const void *scalar, av_int64 elements) noexcept
	{
		const T value = getScalarValue<T>(scalar);
		for (av_int64 i = 0; i < elements; i++)
			dst[i] = src[i] + value;
	}

	template<typename T, typename U>
	void kernel_add_tensors(T *dst, const T *src, U alpha, U beta, BroadcastedDimensions dims) noexcept
	{
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				U tmp = alpha * static_cast<U>(src[i * dims.last + j]);
				if (beta != zero<U>())
					tmp += beta * dst[i * dims.last + j];
				dst[i * dims.last + j] = tmp;
			}
	}

	template<typename dstT, typename srcT, typename biasT>
	void kernel_add_bias(dstT *dst, biasT alpha1, biasT alpha2, const srcT *src, const biasT *bias, biasT beta1, biasT beta2, biasT beta3,
			const dstT *ext, BroadcastedDimensions dims, avActivationType_t type) noexcept
	{
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				biasT input = alpha2 * static_cast<biasT>(src[i * dims.last + j]);

				biasT tmp = input + bias[j];
				if (beta1 != zero<biasT>())
					tmp += beta1 * ext[i * dims.last + j];
				tmp = alpha1 * activation_forward(type, tmp);

				if (beta2 != zero<biasT>())
					tmp += beta2 * ext[i * dims.last + j];
				if (beta3 != zero<biasT>())
					tmp += beta3 * dst[i * dims.last + j];

				dst[i * dims.last + j] = Store<dstT, biasT>::store(tmp);
			}
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace reference;
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
		avStatus_t refScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *alpha,
				const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const av_int64 elements = getTensor(cDesc).volume();
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					kernel_scale_tensor(getPointer<uint8_t>(cMem), getPointer<uint8_t>(aMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT8:
					kernel_scale_tensor(getPointer<int8_t>(cMem), getPointer<int8_t>(aMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT16:
					kernel_scale_tensor(getPointer<int16_t>(cMem), getPointer<int16_t>(aMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT32:
					kernel_scale_tensor(getPointer<int32_t>(cMem), getPointer<int32_t>(aMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT64:
					kernel_scale_tensor(getPointer<int64_t>(cMem), getPointer<int64_t>(aMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_scale_tensor(getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_scale_tensor(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_scale_tensor(getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_scale_tensor(getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha), elements);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					kernel_scale_tensor(getPointer<std::complex<float>>(cMem), getPointer<std::complex<float>>(aMem),
							getAlphaValue<std::complex<float>>(alpha), elements);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					kernel_scale_tensor(getPointer<std::complex<double>>(cMem), getPointer<std::complex<double>>(aMem),
							getAlphaValue<std::complex<double>>(alpha), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem,
				const void *scalar, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const av_int64 elements = getTensor(cDesc).volume();
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					kernel_add_scalar_to_tensor(getPointer<uint8_t>(cMem), getPointer<uint8_t>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_INT8:
					kernel_add_scalar_to_tensor(getPointer<int8_t>(cMem), getPointer<int8_t>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_INT16:
					kernel_add_scalar_to_tensor(getPointer<int16_t>(cMem), getPointer<int16_t>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_INT32:
					kernel_add_scalar_to_tensor(getPointer<int32_t>(cMem), getPointer<int32_t>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_INT64:
					kernel_add_scalar_to_tensor(getPointer<int64_t>(cMem), getPointer<int64_t>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_scalar_to_tensor(getPointer<float16>(cMem), getPointer<float16>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_add_scalar_to_tensor(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_scalar_to_tensor(getPointer<float>(cMem), getPointer<float>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_scalar_to_tensor(getPointer<double>(cMem), getPointer<double>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					kernel_add_scalar_to_tensor(getPointer<std::complex<float>>(cMem), getPointer<std::complex<float>>(aMem), scalar, elements);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					kernel_add_scalar_to_tensor(getPointer<std::complex<double>>(cMem), getPointer<std::complex<double>>(aMem), scalar, elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refAddTensors(avContextDescriptor_t context, const void *alpha, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(aDesc), getTensor(cDesc));
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					kernel_add_tensors(getPointer<uint8_t>(cMem), getPointer<uint8_t>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions);
					break;
				case AVOCADO_DTYPE_INT8:
					kernel_add_tensors(getPointer<int8_t>(cMem), getPointer<int8_t>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions);
					break;
				case AVOCADO_DTYPE_INT16:
					kernel_add_tensors(getPointer<int16_t>(cMem), getPointer<int16_t>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions);
					break;
				case AVOCADO_DTYPE_INT32:
					kernel_add_tensors(getPointer<int32_t>(cMem), getPointer<int32_t>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions);
					break;
				case AVOCADO_DTYPE_INT64:
					kernel_add_tensors(getPointer<int64_t>(cMem), getPointer<int64_t>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_tensors(getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_add_tensors(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_tensors(getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_tensors(getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha), getBetaValue<double>(beta),
							dimensions);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					kernel_add_tensors(getPointer<std::complex<float>>(cMem), getPointer<std::complex<float>>(aMem),
							getAlphaValue<std::complex<float>>(alpha), getBetaValue<std::complex<float>>(beta), dimensions);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					kernel_add_tensors(getPointer<std::complex<double>>(cMem), getPointer<std::complex<double>>(aMem),
							getAlphaValue<std::complex<double>>(alpha), getBetaValue<std::complex<double>>(beta), dimensions);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refAddBias(avContextDescriptor_t context, const void *alpha1, const void *alpha2, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const void *beta1, const void *beta2, const void *beta3, const avMemoryDescriptor_t zMem,
				avActivationType_t activation)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(xDesc), getTensor(bDesc));
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
				{
					switch (getTensor(yDesc).dtype())
					{
						case AVOCADO_DTYPE_INT8:
							kernel_add_bias(getPointer<int8_t>(yMem), getAlphaValue(alpha1), getAlphaValue(alpha2), getPointer<int8_t>(xMem),
									getPointer<float>(bMem), getBetaValue(beta1), getBetaValue(beta2), getBetaValue(beta3), getPointer<int8_t>(zMem),
									dimensions, activation);
							break;
						case AVOCADO_DTYPE_INT32:
							kernel_add_bias(getPointer<int8_t>(yMem), getAlphaValue(alpha1), getAlphaValue(alpha2), getPointer<int32_t>(xMem),
									getPointer<float>(bMem), getBetaValue(beta1), getBetaValue(beta2), getBetaValue(beta3), getPointer<int8_t>(zMem),
									dimensions, activation);
							break;
						default:
							return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
					}
					break;
				}
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_bias(getPointer<float16>(yMem), getAlphaValue(alpha1), getAlphaValue(alpha2), getPointer<float16>(xMem),
							getPointer<float>(bMem), getBetaValue(beta1), getBetaValue(beta2), getBetaValue(beta3), getPointer<float16>(zMem),
							dimensions, activation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_add_bias(getPointer<bfloat16>(yMem), getAlphaValue(alpha1), getAlphaValue(alpha2), getPointer<bfloat16>(xMem),
							getPointer<float>(bMem), getBetaValue(beta1), getBetaValue(beta2), getBetaValue(beta3), getPointer<bfloat16>(zMem),
							dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_bias(getPointer<float>(yMem), getAlphaValue(alpha1), getAlphaValue(alpha2), getPointer<float>(xMem),
							getPointer<float>(bMem), getBetaValue(beta1), getBetaValue(beta2), getBetaValue(beta3), getPointer<float>(zMem),
							dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_bias(getPointer<double>(yMem), getAlphaValue<double>(alpha1), getAlphaValue<double>(alpha2), getPointer<double>(xMem),
							getPointer<double>(bMem), getBetaValue<double>(beta1), getBetaValue<double>(beta2), getBetaValue<double>(beta3),
							getPointer<double>(zMem), dimensions, activation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

