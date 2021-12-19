/*
 * fp16.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */
#include "fp16.hpp"

#include <cstring>
#include <vector>
#include <inttypes.h>

namespace
{
	uint32_t convertmantissa(uint32_t i) noexcept
	{
		uint32_t m = i << 13; // Zero pad mantissa bits
		uint32_t e = 0; // Zero exponent
		while (!(m & 0x00800000))
		{
			e -= 0x00800000;
			m <<= 1;
		}
		m &= ~0x00800000;
		e += 0x38800000;
		return m | e;
	}
	std::vector<uint32_t> init_mantissa_table()
	{
		std::vector<uint32_t> result(2048);
		result[0] = 0;
		uint32_t i = 1;
		for (; i <= 1023; i++)
			result[i] = convertmantissa(i);
		for (; i < 2048; i++)
			result[i] = 0x38000000 + ((i - 1024) << 13);
		return result;
	}
	std::vector<uint32_t> init_exponent_table()
	{
		std::vector<uint32_t> result(64);
		result[0] = 0;
		for (uint32_t i = 1; i <= 30; i++)
			result[i] = i << 23;
		result[31] = 0x47800000;
		result[32] = 0x80000000;
		for (uint32_t i = 33; i <= 62; i++)
			result[i] = 0x80000000 + ((i - 32) << 23);
		result[63] = 0xC7800000;
		return result;
	}
	std::vector<uint16_t> init_offset_table()
	{
		std::vector<uint16_t> result(64, 1024);
		result[0] = 0;
		result[32] = 0;
		return result;
	}
	std::vector<uint16_t> init_base_table()
	{
		std::vector<uint16_t> result(512);
		for (uint32_t i = 0; i < 256; i++)
		{
			int32_t e = i - 127;
			if (e < -24)
			{
				result[i | 0x000] = 0x0000;
				result[i | 0x100] = 0x8000;
			}
			else
			{
				if (e < -14)
				{
					result[i | 0x000] = (0x0400 >> (-e - 14));
					result[i | 0x100] = (0x0400 >> (-e - 14)) | 0x8000;
				}
				else
				{
					if (e <= 15)
					{
						result[i | 0x000] = ((e + 15) << 10);
						result[i | 0x100] = ((e + 15) << 10) | 0x8000;
					}
					else
					{
						if (e < 128)
						{
							result[i | 0x000] = 0x7C00;
							result[i | 0x100] = 0xFC00;
						}
						else
						{
							result[i | 0x000] = 0x7C00;
							result[i | 0x100] = 0xFC00;
						}
					}
				}
			}
		}
		return result;
	}
	std::vector<uint8_t> init_shift_table()
	{
		std::vector<uint8_t> result(512);
		for (uint32_t i = 0; i < 256; i++)
		{
			int32_t e = i - 127;
			if (e < -24)
			{
				result[i | 0x000] = 24;
				result[i | 0x100] = 24;
			}
			else
			{
				if (e < -14)
				{
					result[i | 0x000] = -e - 1;
					result[i | 0x100] = -e - 1;
				}
				else
				{
					if (e <= 15)
					{
						result[i | 0x000] = 13;
						result[i | 0x100] = 13;
					}
					else
					{
						if (e < 128)
						{
							result[i | 0x000] = 24;
							result[i | 0x100] = 24;
						}
						else
						{
							result[i | 0x000] = 13;
							result[i | 0x100] = 13;
						}
					}
				}
			}
		}
		return result;
	}

	float half_to_float(uint16_t h)
	{
		static auto mantissa_table = init_mantissa_table();
		static auto offset_table = init_offset_table();
		static auto exponent_table = init_exponent_table();

		int tmp = mantissa_table[offset_table[h >> 10] + (h & 0x3ff)] + exponent_table[h >> 10];
		float result = 0.0f;
		std::memcpy(&result, &tmp, sizeof(tmp));
		return result;
	}
	uint16_t float_to_half(float f)
	{
		static auto base_table = init_base_table();
		static auto shift_table = init_shift_table();

		uint32_t tmp = 0;
		std::memcpy(&tmp, &f, sizeof(tmp));
		return base_table[(tmp >> 23) & 0x1ff] + ((tmp & 0x007fffff) >> shift_table[(tmp >> 23) & 0x1ff]);
	}

	float bfloat16_to_float(uint16_t x) noexcept
	{
		uint16_t tmp[2] = { 0u, x };
		float result;
		std::memcpy(&result, tmp, sizeof(float));
		return result;
	}
	uint16_t float_to_bfloat16(float x) noexcept
	{
		uint16_t tmp[2];
		std::memcpy(&tmp, &x, sizeof(float));
		return tmp[1];
	}
}
namespace avocado
{
	namespace backend
	{
		float16::float16(unsigned short raw_binary_value) noexcept :
				m_data(raw_binary_value)
		{
		}
		float16::float16(int i) noexcept :
				m_data(float_to_half(static_cast<float>(i)))
		{
		}
		float16::float16(long long i) noexcept :
				m_data(float_to_half(static_cast<float>(i)))
		{
		}
		float16::float16(float f) noexcept :
				m_data(float_to_half(f))
		{
		}
		float16::float16(double f) noexcept :
				m_data(float_to_half(static_cast<float>(f)))
		{
		}
		float16::operator int() const noexcept
		{
			return static_cast<int>(half_to_float(m_data));
		}
		float16::operator float() const noexcept
		{
			return half_to_float(m_data);
		}
		float16::operator double() const noexcept
		{
			return static_cast<double>(half_to_float(m_data));
		}
		float16& float16::operator=(int i) noexcept
		{
			m_data = half_to_float(static_cast<float>(i));
			return *this;
		}
		float16& float16::operator=(float f) noexcept
		{
			m_data = half_to_float(f);
			return *this;
		}
		float16& float16::operator=(double d) noexcept
		{
			m_data = half_to_float(static_cast<float>(d));
			return *this;
		}

		bool operator==(const float16 &lhs, const float16 &rhs) noexcept
		{
			return lhs.m_data == rhs.m_data;
		}
		bool operator!=(const float16 &lhs, const float16 &rhs) noexcept
		{
			return lhs.m_data != rhs.m_data;
		}
		bool operator<(const float16 &lhs, const float16 &rhs) noexcept
		{
			return half_to_float(lhs.m_data) < half_to_float(rhs.m_data);
		}
		bool operator<=(const float16 &lhs, const float16 &rhs) noexcept
		{
			return half_to_float(lhs.m_data) <= half_to_float(rhs.m_data);
		}
		bool operator>(const float16 &lhs, const float16 &rhs) noexcept
		{
			return half_to_float(lhs.m_data) > half_to_float(rhs.m_data);
		}
		bool operator>=(const float16 &lhs, const float16 &rhs) noexcept
		{
			return half_to_float(lhs.m_data) >= half_to_float(rhs.m_data);
		}
		float16 operator&(const float16 &lhs, const float16 &rhs) noexcept
		{
			return float16(lhs.m_data & rhs.m_data);
		}
		float16 operator|(const float16 &lhs, const float16 &rhs) noexcept
		{
			return float16(lhs.m_data & rhs.m_data);
		}
		float16 operator^(const float16 &lhs, const float16 &rhs) noexcept
		{
			return float16(lhs.m_data & rhs.m_data);
		}
		float16 operator~(const float16 &a) noexcept
		{
			return float16(~(a.m_data));
		}

		float16 operator+(const float16 &x) noexcept
		{
			return x;
		}
		float16 operator-(const float16 &x) noexcept
		{
			return float16(-half_to_float(x.m_data));
		}
		float16 operator+(const float16 &lhs, const float16 &rhs) noexcept
		{
			return float16(half_to_float(lhs.m_data) + half_to_float(rhs.m_data));
		}
		float16 operator+(const float16 &lhs, const float &rhs) noexcept
		{
			return float16(half_to_float(lhs.m_data) + rhs);
		}
		float16 operator+(const float &lhs, const float16 &rhs) noexcept
		{
			return float16(lhs + half_to_float(rhs.m_data));
		}

		float16 operator-(const float16 &lhs, const float16 &rhs) noexcept
		{
			return float16(half_to_float(lhs.m_data) - half_to_float(rhs.m_data));
		}
		float16 operator-(const float16 &lhs, const float &rhs) noexcept
		{
			return float16(half_to_float(lhs.m_data) - rhs);
		}
		float16 operator-(const float &lhs, const float16 &rhs) noexcept
		{
			return float16(lhs - half_to_float(rhs.m_data));
		}

		float16 operator*(const float16 &lhs, const float16 &rhs) noexcept
		{
			return float16(half_to_float(lhs.m_data) * half_to_float(rhs.m_data));
		}
		float16 operator*(const float16 &lhs, const float &rhs) noexcept
		{
			return float16(half_to_float(lhs.m_data) * rhs);
		}
		float16 operator*(const float &lhs, const float16 &rhs) noexcept
		{
			return float16(lhs * half_to_float(rhs.m_data));
		}

		float16 operator/(const float16 &lhs, const float16 &rhs) noexcept
		{
			return float16(half_to_float(lhs.m_data) / half_to_float(rhs.m_data));
		}
		float16 operator/(const float16 &lhs, const float &rhs) noexcept
		{
			return float16(half_to_float(lhs.m_data) / rhs);
		}
		float16 operator/(const float &lhs, const float16 &rhs) noexcept
		{
			return float16(lhs / half_to_float(rhs.m_data));
		}

		float16& operator+=(float16 &lhs, const float16 &rhs) noexcept
		{
			lhs = float16(half_to_float(lhs.m_data) + half_to_float(rhs.m_data));
			return lhs;
		}
		float16& operator-=(float16 &lhs, const float16 &rhs) noexcept
		{
			lhs = float16(half_to_float(lhs.m_data) - half_to_float(rhs.m_data));
			return lhs;
		}
		float16& operator*=(float16 &lhs, const float16 &rhs) noexcept
		{
			lhs = float16(half_to_float(lhs.m_data) * half_to_float(rhs.m_data));
			return lhs;
		}
		float16& operator/=(float16 &lhs, const float16 &rhs) noexcept
		{
			lhs = float16(half_to_float(lhs.m_data) / half_to_float(rhs.m_data));
			return lhs;
		}

		float& operator+=(float &lhs, const float16 &rhs) noexcept
		{
			lhs += half_to_float(rhs.m_data);
			return lhs;
		}

		/* bfloat16 implementation */
		bfloat16::bfloat16(unsigned short raw_binary_value) noexcept :
				m_data(raw_binary_value)
		{
		}
		bfloat16::bfloat16(int i) noexcept :
				m_data(float_to_bfloat16(static_cast<float>(i)))
		{
		}
		bfloat16::bfloat16(long long i) noexcept :
				m_data(float_to_bfloat16(static_cast<float>(i)))
		{
		}
		bfloat16::bfloat16(float f) noexcept :
				m_data(float_to_bfloat16(f))
		{
		}
		bfloat16::bfloat16(double f) noexcept :
				m_data(float_to_bfloat16(static_cast<float>(f)))
		{
		}
		bfloat16::operator int() const noexcept
		{
			return static_cast<int>(bfloat16_to_float(m_data));
		}
		bfloat16::operator float() const noexcept
		{
			return bfloat16_to_float(m_data);
		}
		bfloat16::operator double() const noexcept
		{
			return static_cast<double>(bfloat16_to_float(m_data));
		}
		bfloat16& bfloat16::operator=(int i) noexcept
		{
			m_data = bfloat16_to_float(static_cast<float>(i));
			return *this;
		}
		bfloat16& bfloat16::operator=(float f) noexcept
		{
			m_data = bfloat16_to_float(f);
			return *this;
		}
		bfloat16& bfloat16::operator=(double d) noexcept
		{
			m_data = bfloat16_to_float(static_cast<float>(d));
			return *this;
		}

		bool operator==(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return lhs.m_data == rhs.m_data;
		}
		bool operator!=(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return lhs.m_data != rhs.m_data;
		}
		bool operator<(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16_to_float(lhs.m_data) < bfloat16_to_float(rhs.m_data);
		}
		bool operator<=(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16_to_float(lhs.m_data) <= bfloat16_to_float(rhs.m_data);
		}
		bool operator>(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16_to_float(lhs.m_data) > bfloat16_to_float(rhs.m_data);
		}
		bool operator>=(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16_to_float(lhs.m_data) >= bfloat16_to_float(rhs.m_data);
		}
		bfloat16 operator&(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(lhs.m_data & rhs.m_data);
		}
		bfloat16 operator|(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(lhs.m_data & rhs.m_data);
		}
		bfloat16 operator^(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(lhs.m_data & rhs.m_data);
		}
		bfloat16 operator~(const bfloat16 &a) noexcept
		{
			return bfloat16(~(a.m_data));
		}

		bfloat16 operator+(const bfloat16 &x) noexcept
		{
			return x;
		}
		bfloat16 operator-(const bfloat16 &x) noexcept
		{
			return bfloat16(-bfloat16_to_float(x.m_data));
		}
		bfloat16 operator+(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(bfloat16_to_float(lhs.m_data) + bfloat16_to_float(rhs.m_data));
		}
		bfloat16 operator+(const bfloat16 &lhs, const float &rhs) noexcept
		{
			return bfloat16(bfloat16_to_float(lhs.m_data) + rhs);
		}
		bfloat16 operator+(const float &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(lhs + bfloat16_to_float(rhs.m_data));
		}

		bfloat16 operator-(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(bfloat16_to_float(lhs.m_data) - bfloat16_to_float(rhs.m_data));
		}
		bfloat16 operator-(const bfloat16 &lhs, const float &rhs) noexcept
		{
			return bfloat16(bfloat16_to_float(lhs.m_data) - rhs);
		}
		bfloat16 operator-(const float &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(lhs - bfloat16_to_float(rhs.m_data));
		}

		bfloat16 operator*(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(bfloat16_to_float(lhs.m_data) * bfloat16_to_float(rhs.m_data));
		}
		bfloat16 operator*(const bfloat16 &lhs, const float &rhs) noexcept
		{
			return bfloat16(bfloat16_to_float(lhs.m_data) * rhs);
		}
		bfloat16 operator*(const float &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(lhs * bfloat16_to_float(rhs.m_data));
		}

		bfloat16 operator/(const bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(bfloat16_to_float(lhs.m_data) / bfloat16_to_float(rhs.m_data));
		}
		bfloat16 operator/(const bfloat16 &lhs, const float &rhs) noexcept
		{
			return bfloat16(bfloat16_to_float(lhs.m_data) / rhs);
		}
		bfloat16 operator/(const float &lhs, const bfloat16 &rhs) noexcept
		{
			return bfloat16(lhs / bfloat16_to_float(rhs.m_data));
		}

		bfloat16& operator+=(bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			lhs = bfloat16(bfloat16_to_float(lhs.m_data) + bfloat16_to_float(rhs.m_data));
			return lhs;
		}
		bfloat16& operator-=(bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			lhs = bfloat16(bfloat16_to_float(lhs.m_data) - bfloat16_to_float(rhs.m_data));
			return lhs;
		}
		bfloat16& operator*=(bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			lhs = bfloat16(bfloat16_to_float(lhs.m_data) * bfloat16_to_float(rhs.m_data));
			return lhs;
		}
		bfloat16& operator/=(bfloat16 &lhs, const bfloat16 &rhs) noexcept
		{
			lhs = bfloat16(bfloat16_to_float(lhs.m_data) / bfloat16_to_float(rhs.m_data));
			return lhs;
		}

		float& operator+=(float &lhs, const bfloat16 &rhs) noexcept
		{
			lhs += bfloat16_to_float(rhs.m_data);
			return lhs;
		}

	} /* namespace backend */
} /* namespace avocado */
