
#include "AS3.h"
#include <math.h>

typedef struct {
    int n,x,y;
    float *w, *b;
    int nonlin; // 0aucune,1sigm,2tanh,3softmax
    int maxpool; // 01
} LAYER;

LAYER models[] = { {500,32,32,0,0,2,0},{62,1,1,0,0,3,0}, {1000,32,32,0,0,1,0},{1000,1,1,0,0,1,0},{1000,1,1,0,0,1,0},{62,1,1,0,0,1,0} };
int mod_i[] = {0,2,6};

LAYER *L;
int nl;

float *data, *o;
#define BUF_SIZE 20000
    
static AS3_Val initparam(void* self, AS3_Val args) {
	AS3_Val	tmp = AS3_Undefined();
    int il, len;

    for (il=0; il < sizeof(models)/sizeof(LAYER); ++il) {
        tmp = AS3_Get(args, AS3_Int(2*il));
        len = AS3_IntValue(AS3_GetS(tmp, "length"));
        models[il].w = (float*) malloc(len);
        AS3_ByteArray_readBytes(models[il].w, tmp, len);

        tmp = AS3_Get(args, AS3_Int(2*il+1));
        len = AS3_IntValue(AS3_GetS(tmp, "length"));
        models[il].b = (float*) malloc(len);
        AS3_ByteArray_readBytes(models[il].b, tmp, len);
    }
    
    data = (float*) malloc(BUF_SIZE);
    o = (float*) malloc(BUF_SIZE);
        
	return AS3_Int(0);
}

static AS3_Val choosemodel(void* self, AS3_Val args) {
    int il;

	AS3_ArrayValue( args, "IntType", &il );

	L = models + mod_i[il];
	nl = mod_i[il+1] - mod_i[il];
   
	return AS3_Int(0);
}

static AS3_Val prediction(void* self, AS3_Val args) {
	AS3_Val	in_arr = AS3_Undefined(), out_arr = AS3_Array(0);
	float *tmp, d, e;
	int i,j,k,l, n,x,y, newx,newy, il, dx,dy;
	LAYER *pL;

	AS3_ArrayValue( args, "AS3ValType", &in_arr );

	for(i=0; i < 1024; ++i)
    	data[i] = AS3_IntValue(AS3_Get(in_arr, AS3_Int(4*i+1))) /255.0;
    	
    n = 1;
    x = 32;
    y = 32;
    
    #define DATA(l,j,i) data[((l)*y + (j))*x + (i)]
    #define O(k,dy,dx) o[((k)*newy + (dy))*newx + (dx)]
    #define W(k,l,j,i) pL->w[(((k)*n + (l))*pL->y + (j))*pL->x + (i)]
    
    for (il=0; il < nl; ++il) {
        flyield();
        pL = L+il;
        newx = x+1-pL->x;
        newy = y+1-pL->y;

        for (dx=0; dx < newx; ++dx)
        for (dy=0; dy < newy; ++dy)
        for (k=0; k < pL->n; ++k) {
            d = pL->b[k];
            for (l=0; l < n; ++l)
            for(j=0; j < pL->y; ++j)
            for(i=0; i < pL->x; ++i)
                d += DATA(l,j+dy,i+dx)*W(k,l,j,i);
            O(k,dy,dx) = d;
        }

        if(pL->maxpool) {
            for (k=0; k < pL->n; ++k)
            for (dx=0; dx < newx; dx+=2)
            for (dy=0; dy < newy; dy+=2) {
                d=O(k,dy,dx);
                e=O(k,dy,dx+1); if(e>d) d=e;
                e=O(k,dy+1,dx); if(e>d) d=e;
                e=O(k,dy+1,dx+1); if(e>d) d=e;
                O(k,dy/2,dx/2)=d;
            }
            newx /= 2;
            newy /= 2;
        }

        for (dx=0; dx < newx; ++dx)
        for (dy=0; dy < newy; ++dy) {
            e = 0;
            for (k=0; k < pL->n; ++k) {
                d = O(k,dy,dx);
                if(pL->nonlin==1) d=1.0/(1.0 + exp(-d));
                else if(pL->nonlin==2) d=tanh(d);
                else if(pL->nonlin==3) { d=exp(d); e += d; }
                O(k,dy,dx) = d;
            }
            if(pL->nonlin==3 && e)
            for (k=0; k < pL->n; ++k)
                O(k,dy,dx) /= e;
        }
        
        tmp = data;
        data = o;
        o = tmp;
        
        x = newx;
        y = newy;
        n = pL->n;
    }

	for(i=0; i < n*x*y; ++i)
        AS3_Set(out_arr, AS3_Int(i), AS3_Number(data[i]));

	return out_arr;
}

int main() {
	AS3_Val initparamMethod = AS3_Function( NULL, initparam );
	AS3_Val choosemodelMethod = AS3_Function( NULL, choosemodel );
	AS3_Val predictionMethod = AS3_FunctionAsync( NULL, prediction );
	
	AS3_Val result = AS3_Object( "initparam: AS3ValType, choosemodel: AS3ValType, prediction: AS3ValType", initparamMethod, choosemodelMethod, predictionMethod );

	AS3_Release( initparamMethod );
	AS3_Release( choosemodelMethod );
	AS3_Release( predictionMethod );
	
	AS3_LibInit( result );

	return 0;
}
