/*
 * winograd_transforms.cpp
 *
 *  Created on: Feb 19, 2022
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

#include "utils.hpp"
#include "activations.hpp"

#include <map>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::reference;

	template<typename T>
	class matrix
	{
		private:
			std::vector<T> m_data;
			int m_rows = 0;
			int m_cols = 0;
		public:
			matrix() = default;
			matrix(int rows, int cols) :
					m_data(rows * cols),
					m_rows(rows),
					m_cols(cols)
			{
				assert(rows >= 0);
				assert(cols >= 0);
			}
			constexpr matrix(std::initializer_list<std::initializer_list<T>> data) :
					m_rows(data.size()),
					m_cols(data.begin()[0].size())
			{
				assert(m_rows >= 0);
				assert(m_cols >= 0);
				for (int i = 0; i < m_rows; i++)
					for (int j = 0; j < m_cols; j++)
					{
						assert(data.begin()[i].size() == data.begin()[0].size());
						m_data.push_back(data.begin()[i].begin()[j]);
					}
			}
			int rows() const noexcept
			{
				return m_rows;
			}
			int cols() const noexcept
			{
				return m_cols;
			}
			void clear()
			{
				m_data.assign(m_data.size(), T { });
			}
			void fill(T value)
			{
				m_data.assign(m_data.size(), value);
			}
			const T& at(int row, int col) const noexcept
			{
				assert(row >= 0 && row < rows());
				assert(col >= 0 && col < cols());
				return m_data[row * cols() + col];
			}
			T& at(int row, int col) noexcept
			{
				assert(row >= 0 && row < rows());
				assert(col >= 0 && col < cols());
				return m_data[row * cols() + col];
			}
	};

	template<typename T, typename U>
	matrix<T> cast_to(const matrix<U> &x)
	{
		matrix<T> result(x.rows(), x.cols());
		for (int i = 0; i < x.rows(); i++)
			for (int j = 0; j < x.cols(); j++)
				result.at(i, j) = static_cast<T>(x.at(i, j));
		return result;
	}
	template<typename T>
	matrix<T> transpose(const matrix<T> &x)
	{
		matrix<T> result(x.cols(), x.rows());
		for (int i = 0; i < x.rows(); i++)
			for (int j = 0; j < x.cols(); j++)
				result.at(j, i) = x.at(i, j);
		return result;
	}
	template<typename T, typename U = T>
	matrix<T> mult(const matrix<T> &lhs, const matrix<T> &rhs)
	{
		assert(lhs.cols() == rhs.rows());
		matrix<T> result(lhs.rows(), rhs.cols());

		for (int i = 0; i < result.rows(); i++)
			for (int j = 0; j < result.cols(); j++)
			{
				U acc = zero<U>();
				for (int k = 0; k < lhs.cols(); k++)
					acc += static_cast<U>(lhs.at(i, k)) * static_cast<U>(rhs.at(k, j));
				result.at(i, j) = static_cast<T>(acc);
			}
		return result;
	}

	enum class TransformType
	{
		WEIGHT,
		INPUT,
		OUTPUT,
		GRADIENT,
		UPDATE
	};
	std::string to_string(TransformType t)
	{
		switch (t)
		{
			case TransformType::WEIGHT:
				return "WEIGHT";
			case TransformType::INPUT:
				return "INPUT";
			case TransformType::OUTPUT:
				return "OUTPUT";
			case TransformType::GRADIENT:
				return "GRADIENT";
			case TransformType::UPDATE:
				return "UPDATE";
			default:
				return "UNKNOWN";
		}
	}

	struct TransformKey
	{
			size_t kernel_size;
			size_t transform_size;
			TransformType type;
			size_t embedding() const noexcept
			{
				return (kernel_size << 32ull) | (transform_size << 8ull) | static_cast<size_t>(type);
			}
			friend bool operator<(const TransformKey &lhs, const TransformKey &rhs) noexcept
			{
				return lhs.embedding() < rhs.embedding();
			}
	};
	class Transforms
	{
			static std::map<TransformKey, matrix<double>> init_transforms()
			{
				const double c16 = 1.0 / 6.0;
				const double c13 = 1.0 / 3.0;
				const double c23 = 2.0 / 3.0;
				const double c43 = 4.0 / 3.0;
				const double c83 = 8.0 / 3.0;

				std::map<TransformKey, matrix<double>> result;

				result.insert( { TransformKey { 3, 2, TransformType::WEIGHT }, matrix<double>( { { 1.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 }, { 1.0, -1.0,
						1.0 }, { 0.0, 0.0, 1.0 } }) });
				result.insert( { TransformKey { 3, 2, TransformType::INPUT }, matrix<double>( { { 1.0, 0.0, -1.0, 0.0 }, { 0.0, 1.0, 1.0, 0.0 }, {
						0.0, -1.0, 1.0, 0.0 }, { 0.0, -1.0, 0.0, 1.0 } }) });
				result.insert(
						{ TransformKey { 3, 2, TransformType::OUTPUT }, matrix<double>( { { 1.0, 0.5, 0.5, 0.0 }, { 0.0, 0.5, -0.5, 1.0 } }) });
				result.insert( { TransformKey { 3, 2, TransformType::GRADIENT }, matrix<double>( { { 1.0, 0.0 }, { 1.0, 1.0 }, { 1.0, -1.0 }, { 0.0,
						1.0 } }) });
				result.insert( { TransformKey { 3, 2, TransformType::UPDATE }, matrix<double>( { { 1.0, 0.5, 0.5, 0.0 }, { 0.0, 0.5, -0.5, 0.0 }, {
						0.0, 0.5, 0.5, 1.0 } }) });

				result.insert( { TransformKey { 3, 4, TransformType::WEIGHT }, matrix<double>( { { 1.0, 0.0, 0.0 }, { c23, c23, c23 }, { c23, -c23,
						c23 }, { c13, c23, c43 }, { c13, -c23, c43 }, { 0.0, 0.0, 2.0 } }) });
				result.insert( { TransformKey { 3, 4, TransformType::INPUT }, matrix<double>( { { 1.0, 0.0, -1.25, 0.0, 0.25, 0.0 }, { 0.0, 1.0, 1.0,
						-0.25, -0.25, 0.0 }, { 0.0, -1.0, 1.0, 0.25, -0.25, 0.0 }, { 0.0, -1.0, -0.5, 1.0, 0.5, 0.0 }, { 0.0, 1.0, -0.5, -1.0, 0.5,
						0.0 }, { 0.0, 1.0, 0.0, -1.25, 0.0, 0.25 } }) });
				result.insert( { TransformKey { 3, 4, TransformType::OUTPUT }, matrix<double>( { { 1.0, 1.0, 1.0, 0.25, 0.25, 0.0 }, { 0.0, 1.0, -1.0,
						0.5, -0.5, 0.0 }, { 0.0, 1.0, 1.0, 1.0, 1.0, 0.0 }, { 0.0, 1.0, -1.0, 2.0, -2.0, 2.0 } }) });
				result.insert( { TransformKey { 3, 4, TransformType::GRADIENT }, matrix<double>( { { 1.0, 0.0, 0.0, 0.0 }, { c23, c23, c23, c23 }, {
						c23, -c23, c23, -c23 }, { c13, c23, c43, c83 }, { c13, -c23, c43, -c83 }, { 0.0, 0.0, 0.0, 2.0 } }) });
				result.insert( { TransformKey { 3, 4, TransformType::UPDATE }, matrix<double>( { { 1.0, 1.0, 1.0, 0.25, 0.25, 0.0 }, { 0.0, 1.0, -1.0,
						0.5, -0.5, 0.0 }, { 0.0, 1.0, 1.0, 1.0, 1.0, 2.0 } }) });

				result.insert(
						{ TransformKey { 5, 2, TransformType::WEIGHT }, matrix<double>(
								{ { 1.0, 0.0, 0.0, 0.0, 0.0 }, { c23, c23, c23, c23, c23 }, { c23, -c23, c23, -c23, c23 },
										{ c16, c13, c23, c43, c83 }, { c16, -c13, c23, -c43, c83 }, { 0.0, 0.0, 0.0, 0.0, 2.0 } }) });
				result.insert( { TransformKey { 5, 2, TransformType::INPUT }, matrix<double>( { { 1.0, 0.0, -1.25, 0.0, 0.25, 0.0 }, { 0.0, 1.0, 1.0,
						-0.25, -0.25, 0.0 }, { 0.0, -1.0, 1.0, 0.25, -0.25, 0.0 }, { 0.0, -1.0, -0.5, 1.0, 0.5, 0.0 }, { 0.0, 1.0, -0.5, -1.0, 0.5,
						0.0 }, { 0.0, 1.0, 0.0, -1.25, 0.0, 0.25 } }) });
				result.insert( { TransformKey { 5, 2, TransformType::OUTPUT }, matrix<double>( { { 1.0, 1.0, 1.0, 0.5, 0.5, 0.0 }, { 0.0, 1.0, -1.0,
						1.0, -1.0, 2.0 } }) });
				result.insert( { TransformKey { 5, 2, TransformType::GRADIENT }, matrix<double>( { { 1.0, 0.0 }, { c23, c23 }, { c23, -c23 }, { c13,
						c23 }, { c13, -c23 }, { 0.0, 1.0 } }) });
				result.insert(
						{ TransformKey { 5, 2, TransformType::UPDATE }, matrix<double>( { { 1.0, 1.0, 1.0, 0.25, 0.25, 0.0 }, { 0.0, 1.0, -1.0, 0.5,
								-0.5, 0.0 }, { 0.0, 1.0, 1.0, 1.0, 1.0, 0.0 }, { 0.0, 1.0, -1.0, 2.0, -2.0, 0.0 }, { 0.0, 1.0, 1.0, 4.0, 4.0, 4.0 } }) });

				return result;
			}
		public:
			template<typename T>
			static matrix<T> get(size_t kernelSize, size_t transformSize, TransformType type)
			{
				static const std::map<TransformKey, matrix<double>> transforms = init_transforms();

				if (transforms.find(TransformKey { kernelSize, transformSize, type }) == transforms.end())
					throw std::logic_error(
							"unknown winograd transform " + std::to_string(kernelSize) + "x" + std::to_string(transformSize) + ":" + to_string(type));

				auto tmp = transforms.find(TransformKey { kernelSize, transformSize, type });
				return cast_to<T>(tmp->second);
			}
	};

	int get_initial_tile_size(int kernelSize, int transformSize, TransformType transformType)
	{
		switch (transformType)
		{
			case TransformType::WEIGHT:
				return kernelSize;
			case TransformType::INPUT:
				return kernelSize + transformSize - 1;
			case TransformType::OUTPUT:
				return transformSize;
			case TransformType::GRADIENT:
				return transformSize;
			case TransformType::UPDATE:
				return kernelSize;
			default:
				return 0;
		}
	}
	int get_final_tile_size(int kernelSize, int transformSize, TransformType transformType)
	{
		switch (transformType)
		{
			case TransformType::WEIGHT:
				return kernelSize + transformSize - 1;
			case TransformType::INPUT:
				return kernelSize + transformSize - 1;
			case TransformType::OUTPUT:
				return kernelSize + transformSize - 1;
			case TransformType::GRADIENT:
				return kernelSize + transformSize - 1;
			case TransformType::UPDATE:
				return kernelSize + transformSize - 1;
			default:
				return 0;
		}
	}
	int get_stride_tile_size(int kernelSize, int transformSize, TransformType transformType)
	{
		switch (transformType)
		{
			case TransformType::WEIGHT:
				return kernelSize;
			case TransformType::INPUT:
				return transformSize;
			case TransformType::OUTPUT:
				return transformSize;
			case TransformType::GRADIENT:
				return transformSize;
			case TransformType::UPDATE:
				return kernelSize;
			default:
				return 0;
		}
	}

	template<typename T>
	void winograd_initial_transform_2d(const std::array<int, 3> &padding, const T paddingValue, const std::array<int, 3> &filter,
			const TensorDescriptor &inputDesc, const MemoryDescriptor &inputMem, const TensorDescriptor &matricesDesc, MemoryDescriptor &matricesMem,
			TransformType transformType, int transformSize, bool invert = false)
	{
		assert(filter[0] == filter[1]);

		const int height = inputDesc.dimension(1);
		const int width = inputDesc.dimension(2);
		const int filter_size = filter[0];

		const int initial_tile_size = get_initial_tile_size(filter_size, transformSize, transformType);
		const int final_tile_size = get_final_tile_size(filter_size, transformSize, transformType);
		const int stride_tile_size = get_stride_tile_size(filter_size, transformSize, transformType);

		const matrix<T> transform_matrix = Transforms::get<T>(filter_size, transformSize, transformType);
		const matrix<T> transposed_matrix = transpose(transform_matrix);

		matrix<T> initial_tile(initial_tile_size, initial_tile_size);

		int tile_index = 0;
		for (int i0 = 0; i0 < inputDesc.dimension(0); i0++)
			for (int i1 = 0; i1 < height; i1 += stride_tile_size)
				for (int i2 = 0; i2 < width; i2 += stride_tile_size)
				{
					for (int i3 = 0; i3 < inputDesc.dimension(3); i3++)
					{
						initial_tile.clear();
						for (int r = 0; r < initial_tile_size; r++)
							for (int c = 0; c < initial_tile_size; c++)
							{
								int x = padding[0] + i1 + r;
								int y = padding[1] + i2 + c;
								int tmp_r = invert ? initial_tile_size - 1 - r : r;
								int tmp_c = invert ? initial_tile_size - 1 - c : c;
								if (x >= 0 and x < height and y >= 0 and y < width)
									initial_tile.at(tmp_r, tmp_c) = inputMem.data<T>()[inputDesc.getIndex( { i0, x, y, i3 })];
								else
									initial_tile.at(tmp_r, tmp_c) = paddingValue;
							}

//						for (int r = 0; r < initial_tile.rows(); r++)
//						{
//							for (int c = 0; c < initial_tile.cols(); c++)
//								std::cout << static_cast<float>(initial_tile.at(r, c)) << " ";
//							std::cout << '\n';
//						}
//						std::cout << "-------------------------------------------\n";

						matrix<T> tmp = mult(transform_matrix, initial_tile);
//						for (int r = 0; r < tmp.rows(); r++)
//						{
//							for (int c = 0; c < tmp.cols(); c++)
//								std::cout << static_cast<float>(tmp.at(r, c)) << " ";
//							std::cout << '\n';
//						}
//						std::cout << "-------------------------------------------\n";

						matrix<T> final_tile = mult(tmp, transposed_matrix);
//						for (int r = 0; r < final_tile.rows(); r++)
//						{
//							for (int c = 0; c < final_tile.cols(); c++)
//								std::cout << static_cast<float>(final_tile.at(r, c)) << " ";
//							std::cout << '\n';
//						}
//						std::cout << "-------------------------------------------\n";
//						exit(0);

						assert(final_tile.rows() == final_tile_size && final_tile.cols() == final_tile_size);
						for (int r = 0; r < final_tile_size; r++)
							for (int c = 0; c < final_tile_size; c++)
								matricesMem.data<T>()[matricesDesc.getIndex( { r * final_tile_size + c, tile_index, i3 })] = final_tile.at(r, c);
					}
					tile_index++;
				}
	}
	template<typename T>
	void winograd_final_transform_2d(const std::array<int, 3> &filter, const TensorDescriptor &matricesDesc, const MemoryDescriptor &matricesMem,
			const TensorDescriptor &outputDesc, MemoryDescriptor &outputMem, TransformType transformType, int transformSize)
	{
		assert(filter[0] == filter[1]);

		const int height = outputDesc.dimension(1);
		const int width = outputDesc.dimension(2);
		const int filter_size = filter[0];

		const int initial_tile_size = get_initial_tile_size(filter_size, transformSize, transformType);
		const int final_tile_size = get_final_tile_size(filter_size, transformSize, transformType);
		const int stride_tile_size = get_stride_tile_size(filter_size, transformSize, transformType);

		matrix<T> transform_matrix = Transforms::get<T>(filter_size, transformSize, transformType);
		matrix<T> transposed_matrix = transpose(transform_matrix);

		matrix<T> final_tile(final_tile_size, final_tile_size);

		int tile_index = 0;
		for (int i0 = 0; i0 < outputDesc.dimension(0); i0++)
			for (int i1 = 0; i1 < height; i1 += stride_tile_size)
				for (int i2 = 0; i2 < width; i2 += stride_tile_size)
				{
					for (int i3 = 0; i3 < outputDesc.dimension(3); i3++)
					{
						final_tile.clear();
						for (int r = 0; r < final_tile_size; r++)
							for (int c = 0; c < final_tile_size; c++)
								final_tile.at(r, c) = matricesMem.data<T>()[matricesDesc.getIndex( { r * final_tile_size + c, tile_index, i3 })];

						matrix<T> tmp = mult(transform_matrix, final_tile);
						matrix<T> initial_tile = mult(tmp, transposed_matrix);
						assert(initial_tile.rows() == initial_tile_size && initial_tile.cols() == initial_tile_size);

						for (int r = 0; r < initial_tile_size; r++)
							for (int c = 0; c < initial_tile_size; c++)
								if ((i1 + r) < height and (i2 + c) < width)
									outputMem.data<T>()[outputDesc.getIndex( { i0, i1 + r, i2 + c, i3 })] = initial_tile.at(r, c);
					}
					tile_index++;
				}
	}

	template<typename T, typename U = T>
	void output_transform_post_action(const MemoryDescriptor &tmpMem, U alpha1, const TensorDescriptor &matricesDesc,
			const MemoryDescriptor &matricesMem, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, const TensorDescriptor &bDesc,
			const MemoryDescriptor &bMem, U alpha2, const TensorDescriptor &zDesc, const MemoryDescriptor &zMem, U beta,
			avActivationType_t activation)
	{
		const bool use_bias = not bMem.isNull();
		const bool use_ext = not zMem.isNull();
		if (use_bias)
		{
			reference::BroadcastedDimensions dimensions = getBroadcastDimensions(yDesc, bDesc);

			for (av_int64 i = 0; i < dimensions.first; i++)
				for (av_int64 j = 0; j < dimensions.last; j++)
				{
					U src = static_cast<U>(tmpMem.data<T>()[i * dimensions.last + j]);
					U bias = bMem.data<U>()[j];
					U ext = use_ext ? static_cast<U>(zMem.data<T>()[i * dimensions.last + j]) : zero<U>();
					U dst = (beta == zero<U>()) ? zero<U>() : static_cast<U>(yMem.data<T>()[i * dimensions.last + j]);

					U tmp = activation_forward(activation, alpha1 * src + bias + alpha2 * ext) + beta * dst;
					yMem.data<T>()[i * dimensions.last + j] = tmp;
				}
		}
		else
		{
			for (av_int64 i = 0; i < yDesc.volume(); i++)
			{
				U src = static_cast<U>(tmpMem.data<T>()[i]);
				U ext = use_ext ? static_cast<U>(zMem.data<T>()[i]) : zero<U>();
				U dst = (beta == zero<U>()) ? zero<U>() : static_cast<U>(yMem.data<T>()[i]);

				U tmp = activation_forward(activation, alpha1 * src + alpha2 * ext) + beta * dst;
				yMem.data<T>()[i] = tmp;
			}
		}

	}

	std::array<int, 3> filter_shape_to_array(const TensorDescriptor &desc)
	{
		switch (desc.nbDims())
		{
			case 3:
				return std::array<int, 3> { desc.dimension(1), 0, 0 };
			case 4:
				return std::array<int, 3> { desc.dimension(1), desc.dimension(2), 0 };
			case 5:
				return std::array<int, 3> { desc.dimension(1), desc.dimension(2), desc.dimension(3) };
			default:
				return std::array<int, 3> { 0, 0, 0 };
		}
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace reference;

		avStatus_t refWinogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc,
				avMemoryDescriptor_t matricesMem)
		{
			const std::array<int, 3> padding = { 0, 0, 0 };
			const std::array<int, 3> filter = filter_shape_to_array(getTensor(wDesc));
			const bool invert = (getConvolution(config).mode == AVOCADO_CROSS_CORRELATION_MODE);
			switch (getTensor(wDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					winograd_initial_transform_2d(padding, zero<float16>(), filter, getTensor(wDesc), getMemory(wMem), getTensor(matricesDesc),
							getMemory(matricesMem), TransformType::WEIGHT, transformSize, invert);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					winograd_initial_transform_2d(padding, zero<bfloat16>(), filter, getTensor(wDesc), getMemory(wMem), getTensor(matricesDesc),
							getMemory(matricesMem), TransformType::WEIGHT, transformSize, invert);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					winograd_initial_transform_2d(padding, zero<float>(), filter, getTensor(wDesc), getMemory(wMem), getTensor(matricesDesc),
							getMemory(matricesMem), TransformType::WEIGHT, transformSize, invert);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					winograd_initial_transform_2d(padding, zero<double>(), filter, getTensor(wDesc), getMemory(wMem), getTensor(matricesDesc),
							getMemory(matricesMem), TransformType::WEIGHT, transformSize, invert);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refWinogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem,
				const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			std::array<int, 3> padding = getConvolution(config).padding;
			std::array<int, 3> filter = filter_shape_to_array(getTensor(wDesc));
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					winograd_initial_transform_2d(padding, getConvolution(config).getPaddingValue<float16>(), filter, getTensor(xDesc),
							getMemory(xMem), getTensor(matricesDesc), getMemory(matricesMem), TransformType::INPUT, transformSize);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					winograd_initial_transform_2d(padding, getConvolution(config).getPaddingValue<bfloat16>(), filter, getTensor(xDesc),
							getMemory(xMem), getTensor(matricesDesc), getMemory(matricesMem), TransformType::INPUT, transformSize);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					winograd_initial_transform_2d(padding, getConvolution(config).getPaddingValue<float>(), filter, getTensor(xDesc), getMemory(xMem),
							getTensor(matricesDesc), getMemory(matricesMem), TransformType::INPUT, transformSize);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					winograd_initial_transform_2d(padding, getConvolution(config).getPaddingValue<double>(), filter, getTensor(xDesc),
							getMemory(xMem), getTensor(matricesDesc), getMemory(matricesMem), TransformType::INPUT, transformSize);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refWinogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const void *alpha1, const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem,
				const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *alpha2, const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta,
				avActivationType_t activation)
		{
			std::array<int, 3> filter = filter_shape_to_array(getTensor(wDesc));
			MemoryDescriptor tmp_mem(getMemory(yMem).size());

			switch (getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					winograd_final_transform_2d<float16>(filter, getTensor(matricesDesc), getMemory(matricesMem), getTensor(yDesc), tmp_mem,
							TransformType::OUTPUT, transformSize);
					output_transform_post_action<float16, float>(tmp_mem, getAlphaValue<float>(alpha1), getTensor(matricesDesc),
							getMemory(matricesMem), getTensor(yDesc), getMemory(yMem), getTensor(bDesc), getMemory(bMem),
							getAlphaValue<float>(alpha2), getTensor(zDesc), getMemory(zMem), getBetaValue<float>(beta), activation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					winograd_final_transform_2d<bfloat16>(filter, getTensor(matricesDesc), getMemory(matricesMem), getTensor(yDesc), tmp_mem,
							TransformType::OUTPUT, transformSize);
					output_transform_post_action<bfloat16, float>(tmp_mem, getAlphaValue<float>(alpha1), getTensor(matricesDesc),
							getMemory(matricesMem), getTensor(yDesc), getMemory(yMem), getTensor(bDesc), getMemory(bMem),
							getAlphaValue<float>(alpha2), getTensor(zDesc), getMemory(zMem), getBetaValue<float>(beta), activation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					winograd_final_transform_2d<float>(filter, getTensor(matricesDesc), getMemory(matricesMem), getTensor(yDesc), tmp_mem,
							TransformType::OUTPUT, transformSize);
					output_transform_post_action<float>(tmp_mem, getAlphaValue<float>(alpha1), getTensor(matricesDesc), getMemory(matricesMem),
							getTensor(yDesc), getMemory(yMem), getTensor(bDesc), getMemory(bMem), getAlphaValue<float>(alpha2), getTensor(zDesc),
							getMemory(zMem), getBetaValue<float>(beta), activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					winograd_final_transform_2d<double>(filter, getTensor(matricesDesc), getMemory(matricesMem), getTensor(yDesc), tmp_mem,
							TransformType::OUTPUT, transformSize);
					output_transform_post_action<double>(tmp_mem, getAlphaValue<double>(alpha1), getTensor(matricesDesc), getMemory(matricesMem),
							getTensor(yDesc), getMemory(yMem), getTensor(bDesc), getMemory(bMem), getAlphaValue<double>(alpha2), getTensor(zDesc),
							getMemory(zMem), getBetaValue<double>(beta), activation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refWinogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem,
				const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			std::array<int, 3> padding = { 0, 0, 0 };
			std::array<int, 3> filter = filter_shape_to_array(getTensor(wDesc));
			switch (getTensor(dyDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					winograd_initial_transform_2d(padding, zero<float16>(), filter, getTensor(dyDesc), getMemory(dyMem), getTensor(matricesDesc),
							getMemory(matricesMem), TransformType::GRADIENT, transformSize);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					winograd_initial_transform_2d(padding, zero<bfloat16>(), filter, getTensor(dyDesc), getMemory(dyMem), getTensor(matricesDesc),
							getMemory(matricesMem), TransformType::GRADIENT, transformSize);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					winograd_initial_transform_2d(padding, zero<float>(), filter, getTensor(dyDesc), getMemory(dyMem), getTensor(matricesDesc),
							getMemory(matricesMem), TransformType::GRADIENT, transformSize);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					winograd_initial_transform_2d(padding, zero<double>(), filter, getTensor(dyDesc), getMemory(dyMem), getTensor(matricesDesc),
							getMemory(matricesMem), TransformType::GRADIENT, transformSize);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refWinogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const void *alpha, const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta,
				const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem)
		{
			std::array<int, 3> filter = filter_shape_to_array(getTensor(dwDesc));
			avMemoryDescriptor_t tmp_desc;
			avStatus_t status = refCreateMemoryDescriptor(&tmp_desc, getMemory(dwMem).size());
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			switch (getTensor(dwDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					winograd_final_transform_2d<float16>(filter, getTensor(matricesDesc), getMemory(matricesMem), getTensor(dwDesc),
							getMemory(tmp_desc), TransformType::UPDATE, transformSize);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					winograd_final_transform_2d<bfloat16>(filter, getTensor(matricesDesc), getMemory(matricesMem), getTensor(dwDesc),
							getMemory(tmp_desc), TransformType::UPDATE, transformSize);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					winograd_final_transform_2d<float>(filter, getTensor(matricesDesc), getMemory(matricesMem), getTensor(dwDesc),
							getMemory(tmp_desc), TransformType::UPDATE, transformSize);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					winograd_final_transform_2d<double>(filter, getTensor(matricesDesc), getMemory(matricesMem), getTensor(dwDesc),
							getMemory(tmp_desc), TransformType::UPDATE, transformSize);
					break;
				default:
					refDestroyMemoryDescriptor(tmp_desc);
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}

			status = refBinaryOp(context, AVOCADO_BINARY_OP_ADD, alpha, dwDesc, tmp_desc, nullptr, dwDesc, dwMem, beta, dwDesc, dwMem);
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;
			return refDestroyMemoryDescriptor(tmp_desc);
		}

	} /* namespace backend */
} /* namespace avocado */

