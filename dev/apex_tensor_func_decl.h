#ifdef _Tensor1D 
    #error "_Tensor1D must not be defined"
#endif
#ifdef _Tensor2D 
    #error "_Tensor2D must not be defined"
#endif
#ifdef _Tensor2D 
    #error "_Tensor2D must not be defined"
#endif
#ifdef _Tensor3D 
    #error "_Tensor3D must not be defined"
#endif
#ifdef _Tensor4D 
    #error "_Tensor4D must not be defined"
#endif
#ifdef _INLINE
    #error "_INLINE must not be defined"
#endif

#ifdef _APEX_TENSOR_CPU_DECL_
    #define _Tensor1D CTensor1D
    #define _Tensor2D CTensor2D
    #define _Tensor3D CTensor3D
    #define _Tensor4D CTensor4D
    #define _INLINE   inline
#else
#ifdef _APEX_TENSOR_GPU_DECL_
    #define _Tensor1D GTensor1D
    #define _Tensor2D GTensor2D
    #define _Tensor3D GTensor3D
    #define _Tensor4D GTensor4D
    #define _INLINE   
#endif
#endif

namespace apex_tensor{
    /*!\brief namespace of all the functions supporting tensor */
    namespace tensor{
        /* \brief allocate space for given tensor */
        _INLINE void alloc_space( _Tensor1D &ts );
        /* \brief allocate space for given tensor */
        _INLINE void alloc_space( _Tensor2D &ts );
        /* \brief allocate space for given tensor */
        _INLINE void alloc_space( _Tensor3D &ts );
        /* \brief allocate space for given tensor */
        _INLINE void alloc_space( _Tensor4D &ts );
        
        /* \brief free space for given tensor */
        _INLINE void free_space( _Tensor1D &ts );
        /* \brief free space for given tensor */
        _INLINE void free_space( _Tensor2D &ts );
        /* \brief free space for given tensor */
        _INLINE void free_space( _Tensor3D &ts );
        /* \brief free space for given tensor */
        _INLINE void free_space( _Tensor4D &ts );

        /* \brief copy data from one tensor to another */
        _INLINE void copy( _Tensor1D &dst, const CTensor1D &src );
        /* \brief copy data from one tensor to another */
        _INLINE void copy( _Tensor2D &dst, const CTensor2D &src );
        /* \brief copy data from one tensor to another */
        _INLINE void copy( _Tensor3D &dst, const CTensor3D &src );
        /* \brief copy data from one tensor to another */
        _INLINE void copy( _Tensor4D &dst, const CTensor4D &src );        
        
        /* \brief copy data from one tensor to another */
        void copy( _Tensor1D &dst, const GTensor1D &src );
        /* \brief copy data from one tensor to another */
        void copy( _Tensor2D &dst, const GTensor2D &src );
        /* \brief copy data from one tensor to another */
        void copy( _Tensor3D &dst, const GTensor3D &src );
        /* \brief copy data from one tensor to another */
        void copy( _Tensor4D &dst, const GTensor4D &src );
    };
    
    namespace tensor{
        template<typename ST,typename OP>
        _INLINE void scalar_map( _Tensor1D &dst, const _Tensor1D &src, double scalar );
        template<typename ST,typename OP>
        _INLINE void scalar_map( _Tensor2D &dst, const _Tensor2D &src, double scalar );
        template<typename ST,typename OP>
        _INLINE void scalar_map( _Tensor3D &dst, const _Tensor3D &src, double scalar );
        template<typename ST,typename OP>
        _INLINE void scalar_map( _Tensor4D &dst, const _Tensor4D &src, double scalar );

        template<typename ST,typename OP>
        _INLINE void scalar_map( _Tensor1D &dst, const _Tensor1D &src, double scalar );
        template<typename ST,typename OP>
        _INLINE void scalar_map( _Tensor2D &dst, const _Tensor2D &src, double scalar );
        template<typename ST,typename OP>
        _INLINE void scalar_map( _Tensor3D &dst, const _Tensor3D &src, double scalar );
        template<typename ST,typename OP>
        _INLINE void scalar_map( _Tensor4D &dst, const _Tensor4D &src, double scalar );
    };    
};

#undef _Tensor1D 
#undef _Tensor2D 
#undef _Tensor3D 
#undef _Tensor4D 
#undef _INLINE   


