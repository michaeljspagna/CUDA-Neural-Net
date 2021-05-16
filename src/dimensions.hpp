#ifndef DIMENSIONS_HPP
#define DIMENSIONS_HPP

    struct Dimensions
    {
        size_t x, y, z;

        Dimensions(size_t x=0, size_t y=0)
            :x(x), y(y){}
    };

#endif /* DIMENSIONS_HPP */