#ifndef DIMENSION_HPP
#define DIMENSION_HPP
    #include <cstddef>
    struct Dimension
    {
        public:
            std::size_t x, y;

            Dimension(std::size_t x_dimension=0, std::size_t y_dimension=0)
                : x(x_dimension), y(y_dimension){}
    };

#endif /* DIMENSIONS_HPP */