#include<iostream>
#include <cstring>

using namespace std;
//原代码 By zy
//添加   By Rookie 
//添加内容  BigInt 的比较（==,>,<,>=,<=） 
//2010-07-19
  
const int BI_n=100;
struct BigInt
{
  private:
    int a[BI_n];
    mutable int n;
    inline void trunc()const{for(--n;n>=0 && !a[n];--n);++n;}
    void carry(int start=0)
    {
      int i, tmp=0;
      for(i=start;i<n || tmp;++i)
      {
        tmp+=a[i]; a[i]=tmp%10000; tmp/=10000;
      }
      if(i>=n) n=i;
    }
  public:
    BigInt(int x=0)
    {
      memset(a, 0, sizeof(a));
      for(n=0;x;++n) a[n]=x%10000, x/=10000;
    }
    BigInt(const string& x)
    {
      memset(a, 0, sizeof(a)); n=0;
      int block=0, p10=1;
      for(int l=(int)x.size(), i=l-1;i>=0;--i)
      {
        block+=(x[i]-'0')*p10;  p10*=10;
        if(p10==10000)
        {
          a[n++]=block;
          block=0; p10=1;
        }
      }
      if(block) a[n++]=block;
    }
    void output()
    {
      printf("%d",a[n-1]);
      for(int i=n-2;i>=0;--i) printf("%04d",a[i]);
      printf("\n");
      return ;
    }
    friend ostream& operator<<(ostream& out, BigInt x)
    {
      if(x.n==0) return out<<"0";
      out<<x.a[x.n-1];
      for(int i=x.n-2;i>=0;--i)
        out.fill('0'), out.width(4), out<<x.a[i];
      return out;
    }
    
    friend BigInt operator+(const BigInt& x, const BigInt& y)
    {
      BigInt ret;
      ret.n=max(x.n, y.n)+1;
      int tmp=0;
      for(int i=0;i<ret.n;++i)
      {
        if(i<x.n) tmp+=x.a[i];
        if(i<y.n) tmp+=y.a[i];
        ret.a[i]=tmp%10000;  tmp/=10000;
      }
      ret.trunc();
      return ret;
    }
    friend BigInt operator-(const BigInt& x, const BigInt& y)
    {
      BigInt ret;  ret.n=x.n;
      int tmp=0, borrow=0;
      for(int i=0;i<x.n;++i)
      {
        tmp=x.a[i]-y.a[i]-borrow;
        if(tmp<0) ret.a[i]=tmp+10000, borrow=1;
        else      ret.a[i]=tmp,  borrow=0;
      }
      ret.trunc();
      return ret;
    } 
    friend BigInt operator*(const BigInt& x, const BigInt& y)
    {
      BigInt ret;
      ret.n=x.n+y.n;
      for(int i=0;i<x.n;++i)
        for(int j=0;j<y.n;++j)
        {
          ret.a[i+j]+=x.a[i]*y.a[j];
          if(ret.a[i+j]>100000000) ret.carry(i+j);
        }
      ret.carry(0);
      ret.trunc();
      return ret;
    }
};
BigInt a=172654,b=12048;

int main()
{
	cout<<a<<endl;
        a.output();
	cout<<b<<endl;
        b.output();
	return 0;

}
