/*
 * conversions.cpp
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
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refChangeTypeHost(avContextDescriptor_t context, void *dst, avDataType_t dstType, const void *src, avDataType_t srcType,
				avSize_t elements)
		{
			switch (dstType)
			{
				case AVOCADO_DTYPE_UINT8:
					convert_helper(reinterpret_cast<uint8_t*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_INT8:
					convert_helper(reinterpret_cast<int8_t*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_INT16:
					convert_helper(reinterpret_cast<int16_t*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_INT32:
					convert_helper(reinterpret_cast<int32_t*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_INT64:
					convert_helper(reinterpret_cast<int64_t*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					convert_helper(reinterpret_cast<float16*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					convert_helper(reinterpret_cast<bfloat16*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					convert_helper(reinterpret_cast<float*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					convert_helper(reinterpret_cast<double*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					convert_helper(reinterpret_cast<std::complex<float>*>(dst), src, elements, srcType);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					convert_helper(reinterpret_cast<std::complex<double>*>(dst), src, elements, srcType);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refChangeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType, const avMemoryDescriptor_t src,
				avDataType_t srcType, avSize_t elements)
		{
			return refChangeTypeHost(context, reference::getPointer<uint8_t>(dst), dstType, reference::getPointer(src), srcType, elements);
		}
	} /* namespace backend */
} /* namespace avocado */

