#ifndef _APEX_CFRBM_APAPTER_CPP_
#define _APEX_CFRBM_APAPTER_CPP_
#include<vector>

using namespace std;
using namespace apex_tensor;

namespace apex_rbm{

	const int user_num = 1000;

	inline vector<CSTensor2D>	adapt( FILE *f ){
			int user_id = -1, movie_id = -1;
			int rate = -1;
			int old_user_id = -1;
			char other[20];
			vector<CSTensor2D> v;
			int tmp_col = 0;
			int max_rate = 0, movie_count = 0;
			vector<int> movie_rate;

			//first scan get the max rate (maybe 5) and the number 
			//of movies each user rates
			while(fscanf(f, "%d\t%d\t%d\t%s\n", &user_id, &movie_id, &rate, other ) != -1){
				if(rate > max_rate) max_rate = rate;
				if(old_user_id != user_id){
					old_user_id = user_id;
					movie_rate.push_back( movie_count );
					movie_count = 0;
				}
				movie_count ++;
			}
			movie_rate.push_back( movie_count );

			rewind( f );
			//second scan set up the sparce tensor
			//and the vector of the rbm process
			while(fscanf(f, "%d\t%d\t%d\t%s\n", &user_id, &movie_id, &rate, other ) != -1){
				if(old_user_id != user_id){
					old_user_id = user_id;
					tmp_col = 0;
					CSTensor2D tmp_tensor; 
					tmp_tensor.set_param( max_rate, movie_rate[ user_id ] );
					apex_tensor::tensor::alloc_space( tmp_tensor );
					v.push_back(tmp_tensor);
				}
				CSTensor2D tmp_tensor = v.back();
				tmp_tensor.index[tmp_col] = movie_id;
				for(int i = 0; i < max_rate; ++ i){
					tmp_tensor[i][tmp_col] = (i == rate)? 1:0;
				}
				tmp_col ++;
			}
			return v;
	}

};

#endif
