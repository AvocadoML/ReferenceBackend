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
				kernel_convert(dst, reinterpret_cast<const uint8_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_INT8:
				kernel_convert(dst, reinterpret_cast<const int8_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_INT16:
				kernel_convert(dst, reinterpret_cast<const int16_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_INT32:
				kernel_convert(dst, reinterpret_cast<const int32_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_INT64:
				kernel_convert(dst, reinterpret_cast<const int64_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT16:
				kernel_convert(dst, reinterpret_cast<const float16*>(src), elements);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_convert(dst, reinterpret_cast<const bfloat16*>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_convert(dst, reinterpret_cast<const float*>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_convert(dst, reinterpret_cast<const double*>(src), elements);
				break;
			case AVOCADO_DTYPE_COMPLEX32:
				kernel_convert(dst, reinterpret_cast<const std::complex<float>*>(src), elements);
				break;
			case AVOCADO_DTYPE_COMPLEX64:
				kernel_convert(dst, reinterpret_cast<const std::complex<double>*>(src), elements);
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

	template<typename T>
	T op_tensor(avBinaryOp_t operation, T lhs, T rhs) noexcept
	{
		switch (operation)
		{
			case AVOCADO_BINARY_OP_ADD:
				return lhs + rhs;
			case AVOCADO_BINARY_OP_ADD_SQUARE:
				return lhs + square(rhs);
			case AVOCADO_BINARY_OP_SUB:
				return lhs - rhs;
			case AVOCADO_BINARY_OP_MUL:
				return lhs * rhs;
			case AVOCADO_BINARY_OP_DIV:
				return lhs / rhs;
			case AVOCADO_BINARY_OP_MOD:
				return mod(lhs, rhs);
			case AVOCADO_BINARY_OP_POW:
				return pow(lhs, rhs);
			case AVOCADO_BINARY_OP_MIN:
				return std::min(lhs, rhs);
			case AVOCADO_BINARY_OP_MAX:
				return std::max(lhs, rhs);
			case AVOCADO_BINARY_OP_COMPARE_EQ:
				return LogicalOp<T>::value_of(lhs == rhs);
			case AVOCADO_BINARY_OP_COMPARE_NEQ:
				return LogicalOp<T>::value_of(lhs != rhs);
			case AVOCADO_BINARY_OP_COMPARE_GT:
				return LogicalOp<T>::value_of(lhs > rhs);
			case AVOCADO_BINARY_OP_COMPARE_GE:
				return LogicalOp<T>::value_of(lhs >= rhs);
			case AVOCADO_BINARY_OP_COMPARE_LT:
				return LogicalOp<T>::value_of(lhs < rhs);
			case AVOCADO_BINARY_OP_COMPARE_LE:
				return LogicalOp<T>::value_of(lhs <= rhs);
			case AVOCADO_BINARY_OP_LOGICAL_AND:
				return LogicalAnd<T>::value(lhs, rhs);
			case AVOCADO_BINARY_OP_LOGICAL_OR:
				return LogicalOr<T>::value(lhs, rhs);
			case AVOCADO_BINARY_OP_LOGICAL_XOR:
				return LogicalXor<T>::value(lhs, rhs);
		}
		return zero<T>();
	}
	template<typename T, typename U>
	void kernel_op_tensor(T *dst, const T *src1, const T *src2, U alpha1, U alpha2, U beta, BroadcastedDimensions dims, avBinaryOp_t operation)
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
				T result = op_tensor(operation, value1, value2);
				dst[i * dims.last + j] = static_cast<U>(result) + beta * static_cast<U>(dst[i * dims.last + j]);
			}
		}
	}

	template<typename T>
	T single_op_tensor(avUnaryOp_t operation, T x) noexcept
	{
		switch (operation)
		{
			case AVOCADO_UNARY_OP_ABS:
				return avocado::backend::abs(x);
			case AVOCADO_UNARY_OP_CEIL:
				return avocado::backend::ceil(x);
			case AVOCADO_UNARY_OP_COS:
				return avocado::backend::cos(x);
			case AVOCADO_UNARY_OP_EXP:
				return avocado::backend::exp(x);
			case AVOCADO_UNARY_OP_FLOOR:
				return avocado::backend::floor(x);
			case AVOCADO_UNARY_OP_LN:
				return avocado::backend::log(x);
			case AVOCADO_UNARY_OP_NEG:
				return -x;
			case AVOCADO_UNARY_OP_RCP:
				return one<T>() / x;
			case AVOCADO_UNARY_OP_RSQRT:
				return one<T>() / avocado::backend::sqrt(x);
			case AVOCADO_UNARY_OP_SIN:
				return avocado::backend::sin(x);
			case AVOCADO_UNARY_OP_SQUARE:
				return avocado::backend::square(x);
			case AVOCADO_UNARY_OP_SQRT:
				return avocado::backend::sqrt(x);
			case AVOCADO_UNARY_OP_TAN:
				return avocado::backend::tan(x);
			case AVOCADO_UNARY_OP_LOGICAL_NOT:
				return LogicalNot<T>::value(x);
		}
		return zero<T>();
	}
	template<typename T, typename U>
	void kernel_op_single_tensor(T *dst, const T *src, U alpha, U beta, avSize_t elements, avUnaryOp_t operation)
	noexcept
	{
		if (beta == zero<U>())
			clear(dst, elements);

		for (avSize_t i = 0; i < elements; i++)
		{
			T value = static_cast<T>(alpha * static_cast<U>(src[i]));
			T result = single_op_tensor(operation, value);
			dst[i] = static_cast<U>(result) + beta * static_cast<U>(dst[i]);
		}
	}
	template<typename T, typename U>
	void kernel_reduce_tensor(T *dst, const T *src, U alpha, U beta, BroadcastedDimensions dims, avReduceOp_t operation) noexcept
	{
		std::unique_ptr<T[]> workspace = std::make_unique<T[]>(dims.last);

		// first perform initialization of workspace tensor
		for (avSize_t j = 0; j < dims.last; j++)
		{
			switch (operation)
			{
				case AVOCADO_REDUCE_ADD:
				case AVOCADO_REDUCE_AVG:
				case AVOCADO_REDUCE_NORM1:
				case AVOCADO_REDUCE_NORM2:
				case AVOCADO_REDUCE_AMAX:
					workspace[j] = zero<T>();
					break;
				case AVOCADO_REDUCE_MUL:
				case AVOCADO_REDUCE_MUL_NO_ZEROS:
					workspace[j] = one<T>();
					break;
				case AVOCADO_REDUCE_MIN:
					workspace[j] = max_value<T>::get();
					break;
				case AVOCADO_REDUCE_MAX:
					workspace[j] = -max_value<T>::get();
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
					case AVOCADO_REDUCE_ADD:
						workspace[j] += value;
						break;
					case AVOCADO_REDUCE_MUL:
						workspace[j] *= value;
						break;
					case AVOCADO_REDUCE_MIN:
						workspace[j] = std::min(workspace[j], value);
						break;
					case AVOCADO_REDUCE_MAX:
						workspace[j] = std::max(workspace[j], value);
						break;
					case AVOCADO_REDUCE_AMAX:
						workspace[j] = std::max(workspace[j], avocado::backend::abs(value));
						break;
					case AVOCADO_REDUCE_AVG:
						workspace[j] += value;
						break;
					case AVOCADO_REDUCE_NORM1:
						workspace[j] += avocado::backend::abs(value);
						break;
					case AVOCADO_REDUCE_NORM2:
						workspace[j] += square(value);
						break;
					case AVOCADO_REDUCE_MUL_NO_ZEROS:
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
			case AVOCADO_REDUCE_AVG:
				for (avSize_t j = 0; j < dims.last; j++)
					workspace[j] = static_cast<U>(workspace[j]) / dims.first;
				break;
			case AVOCADO_REDUCE_NORM2:
				for (avSize_t j = 0; j < dims.last; j++)
					workspace[j] = avocado::backend::sqrt(static_cast<U>(workspace[j]));
				break;
		}

		if (beta == zero<U>())
			clear(dst, dims.last);
		// now store result in dst
		for (avSize_t j = 0; j < dims.last; j++)
			dst[j] = alpha * static_cast<U>(workspace[j]) + beta * static_cast<U>(dst[j]);
	}
	template<typename T, typename U>
	void kernel_add_tensors(T *dst, const T *src, U alpha1, U alpha2, U beta1, U beta2, BroadcastedDimensions dims, avActivationType_t type) noexcept
	{
		if (beta1 == zero<U>() and beta2 == zero<U>())
			clear(dst, volume(dims));

		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
			{
				U tmp = alpha2 * static_cast<U>(src[j]) + beta2 * static_cast<U>(dst[i * dims.last + j]);
				dst[i * dims.last + j] = Store<T, U>::store(alpha1 * tmp + beta1 * static_cast<U>(dst[i * dims.last + j]));
			}
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refChangeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType, const avMemoryDescriptor_t src,
				avDataType_t srcType, avSize_t elements)
		{
			switch (dstType)
			{
				case AVOCADO_DTYPE_UINT8:
					convert_helper(getPointer<uint8_t>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_INT8:
					convert_helper(getPointer<int8_t>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_INT16:
					convert_helper(getPointer<int16_t>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_INT32:
					convert_helper(getPointer<int32_t>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_INT64:
					convert_helper(getPointer<int64_t>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					convert_helper(getPointer<float16>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					convert_helper(getPointer<bfloat16>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					convert_helper(getPointer<float>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					convert_helper(getPointer<double>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					convert_helper(getPointer<std::complex<float>>(dst), getPointer(src), elements, srcType);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					convert_helper(getPointer<std::complex<double>>(dst), getPointer(src), elements, srcType);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
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
		avStatus_t refBinaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(aDesc), getTensor(bDesc));
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_op_tensor(getPointer<float16>(cMem), getPointer<float16>(aMem), getPointer<float16>(bMem), getAlphaValue(alpha1),
							getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_op_tensor(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getPointer<bfloat16>(bMem), getAlphaValue(alpha1),
							getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_op_tensor(getPointer<float>(cMem), getPointer<float>(aMem), getPointer<float>(bMem), getAlphaValue(alpha1),
							getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_op_tensor(getPointer<double>(cMem), getPointer<double>(aMem), getPointer<double>(bMem), getAlphaValue<double>(alpha1),
							getAlphaValue<double>(alpha2), getBetaValue<double>(beta), dimensions, operation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const avSize_t elements = getTensor(aDesc).volume();
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_op_single_tensor(getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha), getBetaValue(beta), elements,
							operation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_op_single_tensor(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha), getBetaValue(beta),
							elements, operation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_op_single_tensor(getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha), getBetaValue(beta), elements,
							operation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_op_single_tensor(getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha),
							getBetaValue<double>(beta), elements, operation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(aDesc), getTensor(cDesc));
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_reduce_tensor(getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions,
							operation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_reduce_tensor(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions,
							operation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_reduce_tensor(getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions,
							operation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_reduce_tensor(getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha), getBetaValue<double>(beta),
							dimensions, operation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refAddTensors(avContextDescriptor_t context, const void *alpha1, const void *alpha2, const avTensorDescriptor_t bDesc,
				const avMemoryDescriptor_t bMem, const void *beta1, const void *beta2, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				avActivationType_t activation)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(cDesc), getTensor(bDesc));
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
					kernel_add_tensors(getPointer<int8_t>(cMem), getPointer<int8_t>(bMem), getAlphaValue(alpha1), getAlphaValue(alpha2),
							getBetaValue(beta1), getBetaValue(beta2), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_tensors(getPointer<float16>(cMem), getPointer<float16>(bMem), getAlphaValue(alpha1), getAlphaValue(alpha2),
							getBetaValue(beta1), getBetaValue(beta2), dimensions, activation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_add_tensors(getPointer<bfloat16>(cMem), getPointer<bfloat16>(bMem), getAlphaValue(alpha1), getAlphaValue(alpha2),
							getBetaValue(beta1), getBetaValue(beta2), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_tensors(getPointer<float>(cMem), getPointer<float>(bMem), getAlphaValue(alpha1), getAlphaValue(alpha2),
							getBetaValue(beta1), getBetaValue(beta2), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_tensors(getPointer<double>(cMem), getPointer<double>(bMem), getAlphaValue<double>(alpha1),
							getAlphaValue<double>(alpha2), getBetaValue<double>(beta1), getBetaValue<double>(beta2), dimensions, activation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

