//
//  network.c
//  mnist
//
//  Created by TU on 2016/09/28.
//  Copyright © 2016年 TU. All rights reserved.
//

#include "network.h"
#include "main.h"



void matmul(int row1, int col1, double fMat1[row1][col1],
            int row2, int col2, double fMat2[row2][col2],
            int row3, int col3, double tMat3[row3][col3]){
    if (col1 != row2 || row1 != row3 || col2 != col3){
        printf("Mat dimension did not match in matmul\n");
        exit(0);
    }
    
    for (int i=0; i<row3; i++){
        for(int j=0; j<col3; j++){
            for (int k=0; k<col1; k++)
                tMat3[i][j] += fMat1[i][k] * fMat2[k][j];
        }
    }
}

void linCon(int rowx, int colx, double x[rowx][colx],
                 int roww, int colw, double w[roww][colw],
                 int rowb, int colb, double b[rowb][colb],
                 int rowh, int colh, double h[rowh][colh]){
    if (colx != roww || rowx != rowh || colw != colh || colw != rowb || colb != 1){
        printf("Mat shape did not match in linCon\n");
        exit(0);
    }
    int i,j,k;
    for (i =0; i<rowh; i++){
        for(j=0; j<colh; j++){
            for(k=0; k<colx; k++){
                h[i][j] += x[i][k] * w[k][j];
                //printf("i =%d, j= %d, k= %d\n", i,j,k);
                //printf("x[%d][%d] = %f\n", i, k, (double)(x[i][k]));
            }
            //printf("h[%d][%d]= %e\n", i, j, h[i][j]);
            h[i][j] += b[j][0];
            //printf("z[%d][%d]= %e\n", i, j, z[i][j]);
        }
    }
}

void tanh_mat(int rowh, int colh, double h[rowh][colh],
              int rowz, int colz, double z[rowz][colz]){
    int i,j;
    for(i=0; i<rowh; i++){
        for(j=0; j<colh; j++){
            z[i][j] = tanh(h[i][j]);
        }
    }
}

void linCon_softmax(int rowx, int colx, double x[rowx][colx],
                    int roww, int colw, double w[roww][colw],
                    int rowb, int colb, double b[rowb][colb],
                    int rowz, int colz, double z[rowz][colz]){
    if (colx != roww || rowx != rowz || colw != colz || colw != rowb || colb != 1){
        printf("Mat shape did not match in linCon_softmax\n");
        exit(0);
    }
    int i,j,k;
    
    double (*h)[colw];
    h = calloc(rowx*colw, sizeof(double));
    for (i =0; i<rowz; i++){
        for(j=0; j<colz; j++){
            for(k=0; k<colx; k++){
                h[i][j] += x[i][k] * w[k][j];
            }
            //printf("h[%d][%d]= %e\n", i, j, h[i][j]);
            h[i][j] += b[j][0];
        }
    }
    
    for (int i=0; i<rowz; i++){
        double prob[colz];
        double maxh = h[i][0]; /* init  */
        double sum = 0;
        
        for(int j=0; j<colz; j++){
            if (maxh < h[i][j])
                maxh = h[i][j];
        }
        
        for(int j=0; j<colz; j++){
            //printf("atest: %f\n", exp(h[i][j]));
            prob[j] = exp(h[i][j]-maxh);
            sum += prob[j];
        }
        for(int j=0; j<colz; j++){
            z[i][j] = prob[j]/sum;
        }
    }
    free(h);
}

void backprop_tanh(int rowu, int colu, double u[rowu][colu],
                   int roww, int colw, double w[roww][colw],
                   int rowd1, int cold1, double d1[rowd1][cold1],
                   int rowd0, int cold0, double d0[rowd0][cold0]){
    if (cold1 != colw /*(row(w.T)*/ || colu != roww /*col(w.T)*/ || cold0 != roww){
        printf ("Mat shape did not match in backprop_tanh func\n");
        exit(0);
    }
    
    double (*temp)[colw]; /* delta1.dot(w1.T) */
    temp = calloc(rowd1*colw, sizeof(double));
    for (int i=0; i< rowd1; i++ ){
        for(int j=0; j<colw; j++){
            for(int k=0; k<cold1; k++){
                temp[i][j] += d1[i][k] * w[j][k]; /* w.T */
            }
        }
    }
    
    for(int i=0; i< rowd0; i++){
        for(int j=0; j<cold0; j++){
            /* delta0 = tanh'(u0) * delta1.dot(w1) */
            d0[i][j] = (1- pow(tanh(u[i][j]), 2.0)) * temp[i][j];
        }
    }
    free(temp);
}

void refine_variables(int roww, int colw, double w[roww][colw],
                      int rowd, int cold, double d[rowd][cold], /* delta(l)*/
                      int rowz, int colz, double z[rowz][colz], /* z(l-1) */
                      double alpha){
    if(rowz != rowd || roww != colz || colw != cold){
        printf("Mat shape did not match in refine_var func\n");
        exit(0);
    }
    
    /* debug*/
    //printf("w[0][0]= %f\n", w[0][0]);
    
    /* w_refined = (z.T).dot(d) */
    int i,j,k;
    double sum;
    for(i=0; i<roww; i++){
        for(j=0;j<colw; j++){
            sum =0;
            for(k=0; k<rowz; k++){
                sum += z[k][i]*d[k][j];
                //printf("z[%d][%d]:%f    ,d[%d][%d]:%f\n", k,i,z[k][i],k,j,d[k][j]);
            }
            w[i][j] -= alpha/(double)(rowz) * sum;
            //printf("change amount: %f\n",sum);
            //printf("sum[%d][%d]: %f \n", i,j,sum);
        }
    }
    
}

void init(unsigned int range,
          int roww, int colw, double w[roww][colw],
          int rowb, int colb, double b[rowb][colb]){
    if (colw != rowb){
        printf("Matrix shape did not match in init func\n");
        exit(0);
    }
    
    for (int i=0; i<roww; i++){
        for(int j=0; j<colw; j++){
            w[i][j] = ((double)rand()/RAND_MAX - 0.5)*range/2.0;
        }
    }
    
    for (int j =0; j<colw; j++){
        b[j][0] = 0;
    }
}


double cross_entropy(int rowy_, int coly_, double y_[rowy_][coly_],
                     int rowd, int cold, double d[rowd][cold]){
    double e=0;
    
    for(int i=0; i<rowy_; i++){
        for(int j=0; j<coly_; j++){
            e -= d[i][j]*log(y_[i][j]);
        }
    }
    return e;
}

double answer(int rowd, int cold, double d[rowd][cold],
              int rowy_, int coly_, double y_[rowy_][coly_]){
    long ans[cold][cold];
    for(int i=0;i<cold;i++)
        for(int j=0;j<cold;j++)
            ans[i][j] = 0; //initialize
    
    int i,j, good=0;
    for(i=0; i<rowd; i++){
        double maxy_ =0.0;
        int maxd_index=0, maxy__index = 0;
        for(j=0; j<cold; j++){
            double tempd = d[i][j];
            double tempy_ = y_[i][j];
            
            if (maxd_index < d[i][j])
                maxd_index = j;
            if (maxy_ < y_[i][j]){
                maxy__index = j;
                maxy_ = y_[i][j];
            }
        }
        if (maxd_index == maxy__index)
            good++;
        ans[maxy__index][maxd_index]++;
    }
    printf("result for train: %d/%i= %.5f\n", good, i, (double)good/(double)(i));
    
    //print cross table
    printf("      ");
    for(j=0; j<cold; j++)
        printf("%8d", j);
    printf("\n");
    for(i=0; i<cold; i++){
        printf("%8d", i);
        for(j=0; j<cold; j++){
            printf("%8ld", ans[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    return (double)good/(double)(i);
}
