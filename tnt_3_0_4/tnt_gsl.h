#ifdef TNT_GSL_H_
#define TNT_GSL_H_

/*



	To convert from TNT::Vector<double> "x"   to (gsl_vector_double *)  :


	 		&(gsl_vector_view_array( &x[0], x.size()).vector)

		&(glsl_vector_cont_view_aray_with_stride( &x[3], istride, N).vector)

	Example:

		
  in GSL:


		int gsl_linalg_SV_decomp(gsl_matrix_double *A, gsl_matrix_double *V, 
				gsl_vector_double  *S, gsl_vector_double *work)


  becomes in TNT:

		int tnt_gsl_linalg_SV_decomp( Matrix<double> A, Matrix<double> V,
						Vector<double> S)
		{

				Vector<double> work(A.num_cols());

				return gsl_linalg_SV_decomp(  &( gsl_matrix_view_array(&A[0], 
							A.num_rows(), A.num_cols()).vector), .... );

		}
*/



 void foo(

#endif
// TNT_GSL_H_
