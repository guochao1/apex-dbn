#ifndef _APEX_CFRBM_APAPTER_CPP_
#define _APEX_CFRBM_APAPTER_CPP_
#include<vector>
#include<memory>

using namespace std;
using namespace apex_tensor;

namespace apex_rbm{

	const int user_num = 1000;

	inline vector<CSTensor2D>	adapt( FILE *f ){
			size_t user_id = -1, movie_id = -1;
			int rate = -1;
			size_t old_user_id = -1;
			char other[20];
			vector<CSTensor2D> v;
			CSTensor2D* tmp_tensor;
			size_t tmp_col = 0;
			int max_rate = 0, tmp_user = 0, movie_count = 0;
			int movie_rate[ user_num ];
			memset(movie_rate, 0, user_num);
			while(fscanf(f, "%d\t%d\t%d\t%s\n", &user_id, &movie_id, &rate, other )){
				if(rate < max_rate) max_rate = rate;
				if(old_user_id != user_id){
					movie_rate[ tmp_user ++ ] = movie_count;
					movie_count = 0;
				}
				movie_count ++;
			}
			rewind( f );
			while(fscanf(f, "%d\t%d\t%d\t%s\n", &user_id, &movie_id, &rate, other )){
				if(old_user_id != user_id){
					tmp_col = 0;
					tmp_tensor = new CSTensor2D(); 
					tmp_tensor->set_param( max_rate, movie_rate[ user_id - 1 ] );
					apex_tensor::tensor::alloc_space( *tmp_tensor );
					v.push_back(*tmp_tensor);
				}
				tmp_tensor->index[tmp_col++] = movie_id;
				for(int i = 0; i < max_rate; ++ i){
					tmp_tensor->elem[ i*tmp_tensor->x_max] = (i == rate)? 1:0;
				}
			}
			return v;
	}

};

#endif
