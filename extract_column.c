#include<stdio.h>
#include<stdlib.h>




void add(int m[3][3], int b[3], int column_number)
{
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      if(j == column_number)
      {
        b[i++] = m[i][j]
      }
  for (k =0 k<3 ;k++)
  {
      print("%d",b[k])
  }
}



int main()
{
    int a[][3] = {{5,6,7}, {8,9,10}, {3,1,2}};
    int b[3];
    int c[3];
    int d[3];


    extract_column(a,b,1);
    






    return 0;
}