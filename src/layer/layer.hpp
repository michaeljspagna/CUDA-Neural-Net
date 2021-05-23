#ifndef LAYER_HPP
#define LAYER_HPP

    #include <string>
    #include "../util/tensor.hpp"
    /**
     * Base class for all layer classes
     */
    struct Layer
    {
        protected:
            std::string title;
        
        public:
            virtual auto foward_propagate(Tensor& A) -> Tensor& = 0;
            virtual auto back_propagate(Tensor& Z_error, float learning_rate) -> Tensor& = 0;

            auto getTitle() -> std::string 
            {
                return title;
            }
    };
    
#endif /* LAYER_HPP */