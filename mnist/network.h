//
//  network.h
//  mnist
//
//  Created by TU on 2016/09/28.
//  Copyright © 2016年 TU. All rights reserved.
//

#ifndef network_h
#define network_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void matmul(int row1, int col1, double fMat1[row1][col1],
            int row2, int col2, double fMat2[row2][col2],
            int row3, int col3, double tMat3[row3][col3]);
void linCon(int rowx, int colx, double x[rowx][colx],
                 int roww, int colw, double w[roww][colw],
                 int rowb, int colb, double b[rowb][colb],
                 int rowh, int colh, double h[rowh][colh]);
void tanh_mat(int rowh, int colh, double h[rowh][colh],
              int rowz, int colz, double z[rowz][colz]);
void linCon_softmax(int rowx, int colx, double x[rowx][colx],
                    int roww, int colw, double w[roww][colw],
                    int rowb, int colb, double b[rowb][colb],
                    int rowz, int colz, double z[rowz][colz]);
void backprop_tanh(int rowu, int colu, double u[rowu][colu],
                   int roww, int colw, double w[roww][colw],
                   int rowd1, int cold1, double d1[rowd1][cold1],
                   int rowd0, int cold0, double d0[rowd0][cold0]);
void refine_variables(int roww, int colw, double w[roww][colw],
                      int rowd, int cold, double d[rowd][cold], /* delta(l)*/
                      int rowz, int colz, double z[rowz][colz], /* z(l-1) */
                      double alpha);
void init(unsigned int range,
          int roww, int colw, double w[roww][colw],
          int rowb, int colb, double b[rowb][colb]);
double cross_entropy(int rowy_, int coly_, double y_[rowy_][coly_],
                     int rowd, int cold, double d[rowd][cold]);
double answer(int rowd, int cold, double d[rowd][cold],
              int rowy_, int coly_, double y_[rowy_][coly_]);

#endif /* network_h */
