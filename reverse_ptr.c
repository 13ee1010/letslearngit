#include<stdio.h>

#define MAX_SIZE 100
int main()
{   
    int arr[MAX_SIZE];
    int *left = arr;
    int number;
    int *right;
    int tmp;
    printf("enter number of elements\n");
    scanf("%d",&number);

    right = arr[number - 1];

    printf("enter elements\n");
    for(int i = 0 ; i < number; i++)
    {
        scanf("%d", left);
        left++;
    }
    left = arr;
    while(left <= right)
    {   
        tmp = *left;
        *left = *right;
        *left = tmp;
        left++;
        right--;

    }

    for(int i = 0 ; i< number ; i++)
    {
        printf("%d\n",*arr);
        arr++;
    }
    



    



    return 0;
}