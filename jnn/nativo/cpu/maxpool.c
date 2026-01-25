#include "maxpool.h"
#include <stdio.h>

void cpu_maxpool2d_forward(const maxpool2d_fwd_params_t* params) {
    const double* restrict X = params->X;
    double* Y = params->Y;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_pool = params->alt_pool;
    const int larg_pool = params->larg_pool;
    const int alt_std = params->alt_std;
    const int larg_std = params->larg_std;

    const int lotes = params->lotes;
    const int canais = params->canais;

    const int alt_y = (alt_x - alt_pool) / alt_std + 1;
    const int larg_y = (larg_x - larg_pool) / larg_std + 1;

    const int area_x = alt_x * larg_x;
    const int area_y = alt_y * larg_y;

    const int std_lote_x = canais * area_x;
    const int std_lote_y = canais * area_y;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < lotes; b++) {
        for (int c = 0; c < canais; c++) {

            const int off_x_bc = b * std_lote_x + c * area_x;
            const int off_y_bc = b * std_lote_y + c * area_y;

            for (int i = 0; i < alt_y; i++) {
                const int base_x_h = off_x_bc + i * alt_std * larg_x;
                const int base_y_h = off_y_bc + i * larg_y;

                for (int j = 0; j < larg_y; j++) {
                    double max = MIN_DOUBLE_VAL;
                    const int base_x_w = base_x_h + j * larg_std;

                    for (int ph = 0; ph < alt_pool; ph++) {
                        const int lin_x = base_x_w + ph * larg_x;
                        for (int pw = 0; pw < larg_pool; pw++) {
                            double v = X[lin_x + pw];
                            if (v > max) max = v;
                        }
                    }

                    Y[base_y_h + j] = max;
                }
            }
        }
    }
}

void cpu_maxpool2d_backward(const maxpool2d_bwd_params_t* params) {
    const double* restrict X  = params->X;
    const double* restrict GS = params->GS;
    double* restrict GE = params->GE;

    const int lotes = params->lotes;
    const int canais = params->canais;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x; 
    const int alt_gs = params->alt_gs;
    const int larg_gs = params->larg_gs;

    const int alt_pool = params->alt_pool;
    const int larg_pool = params->larg_pool;
    const int alt_std = params->alt_std;
    const int larg_std = params->larg_std;

    const int area_x  = alt_x  * larg_x;
    const int area_gs = alt_gs * larg_gs;

    const int bloco_e  = canais * area_x;
    const int bloco_gs = canais * area_gs;

   #pragma omp parallel for schedule(static)
    for (int bc = 0; bc < lotes * canais; bc++) {
        const int b = bc / canais;
        const int c = bc % canais;
        const int base_x  = b * bloco_e  + c * area_x;
        const int base_gs = b * bloco_gs + c * area_gs;
        const int base_ge = b * bloco_e  + c * area_x;

        for (int i = 0; i < alt_gs; i++) {
            const int lin_ini = i * alt_std;
            const int lin_fim = (lin_ini + alt_pool < alt_x) ? (lin_ini + alt_pool) : alt_x;

            for (int j = 0; j < larg_gs; j++) {
                const int col_ini = j * larg_std;
                const int col_fim = (col_ini + larg_pool < larg_x) ? (col_ini + larg_pool) : larg_x;

                double val_max = MIN_DOUBLE_VAL;
                int lin_max = lin_ini;
                int col_max = col_ini;

                for (int y = lin_ini; y < lin_fim; y++) {
                    const int linha = base_x + y * larg_x;
                    for (int x = col_ini; x < col_fim; x++) {
                        const double v = X[linha + x];
                        if (v > val_max) {
                            val_max = v;
                            lin_max = y;
                            col_max = x;
                        }
                    }
                }

                GE[base_ge + lin_max * larg_x + col_max] +=
                    GS[base_gs + i * larg_gs + j];
            }
        }
    }
}