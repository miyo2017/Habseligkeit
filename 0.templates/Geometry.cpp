#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <queue>
#include <map>
#include <list>
using namespace std;

template <typename T>
inline void upd_max(T &dest,const T& src){if(dest<src)dest=src;return ;}
template <typename T>
inline void upd_min(T &dest,const T& src){if(dest>src)dest=src;return ;}

#define sqr(x) ((x)*(x))
#define fi frist
#define se second

const double Eps=1e-6;
const double PI=acos(-1.0);

inline int sig(double x, double eps = Eps)
{
  return x<-eps?-1:x>eps;
}

inline double deg2rad(double d)
{
  return d*PI/180.0;
}
inline double rad2deg(double r)
{
  return r*180.0/PI;
}

struct Point
{
  double x,y;
  Point(){};
  Point(double x,double y):x(x),y(y){};
  Point operator + (const Point& b)
  {
    return Point(x+b.x,y+b.y);
  }
  Point operator - (const Point& b)
  {
    return Point(x-b.x,y-b.y);
  }
  Point operator * (const double& a)
  {
    return Point(x*a,y*a);
  }
  Point operator / (const double& a)
  {
    return Point(x/a,y/a);
  }
  double operator * (const Point& b)
  {
    return x*b.x+y*b.y;
  }
  double operator % (const Point& b)
  {
    return x*b.y-y*b.x;
  }
  double dis()
  {
    return hypot(x,y);
  }
  double dis2()
  {
    return sqr(x)+sqr(y);
  }
  double alpha() 
  {
    return atan2(y,x);
  }
  double disTo(const Point& a) 
  {
    double dx=x-a.x,dy=y-a.y;
    return hypot(dx,dy);
  }
  double alphaTo(const Point& a)
  {
    double dx=x-a.x,dy=y-a.y;
    return atan2(dy,dx);
  }
  Point rot90()
  {
    return Point(y,-x);
  }
  Point rot(double al)//radian measure---- counter-clockwise
  {
    return Point(x*cos(al)-y*sin(al),x*sin(al)+y*cos(al));
  }
};
const Point O=Point(0,0);

typedef vector<Point> vP;

//三角形的外心，重心
Point ccenter(Point p1,Point p2,Point p3)
{
  Point ret;
  ret=(p3/p3.dis())*((p3-p2)*(p1-p3))/((p2-p1)%(p3-p2));
  ret=p1+p2-ret;
  ret=ret/2;
  return ret;
}
//点到直线的距离
double disLP(Point p1,Point p2,Point q)
{
  return abs((p2-p1)%(q-p1))/(p2-p1).dis();
}

//点到线段的距离
double disSP(Point p1,Point p2,Point q)
{
  if(((p2-p1)*(q-p1))<Eps)return (q-p1).dis();
  if(((p1-p2)*(q-p2))<Eps)return (q-p2).dis();
    return disLP(p1,p2,q);
}

//线段与线段相交
bool crsSS(Point p1,Point p2,Point q1,Point q2) 
{
  if (max(p1.x,p2.x)+Eps<min(q1.x, q2.x)) return false;
  if (max(q1.x,q2.x)+Eps<min(p1.x, p2.x)) return false;
  if (max(p1.y,p2.y)+Eps<min(q1.y, q2.y)) return false;
  if (max(q1.y,q2.y)+Eps<min(p1.y, p2.y)) return false;
  return sig((p2-p1)%(q1-p1))*sig((p2-p1)%(q2-p1))<Eps 
      && sig((q2-q1)%(p1-q1))*sig((q2-q1)%(p2-q1))<Eps;
}
//线段与圆相交
bool crsCS(Point c,double r,Point p1,Point p2)  
{
  return disSP(p1,p2,c)<r+Eps &&
      (r<(c-p1).dis()+Eps||r<(c-p2).dis()+Eps);
}
//圆与圆相交
bool crsCC(Point c1,double r1,Point c2,double r2) 
{
  double dis=(c1-c2).dis();
  return dis<r1+r2+Eps && abs(r1-r2)<dis+Eps;
}
//点与直线的垂足
Point proj(Point p1,Point p2,Point q) 
{
  return p1+((p2-p1)*((p2-p1)*((q-p1))/(p2-p1).dis2()));
}
//直线与直线平行
bool isLLP(Point p1,Point p2,Point q1,Point q2)
{
  return sig((q2-q1)%(p2-p1))==0;
}
//直线与直线的交点(先判平行）
Point isLL(Point p1,Point p2,Point q1,Point q2) 
{
  double d=(q2-q1)%(p2-p1);
  return p1+((p2-p1)*((q2-q1)%(q1-p1))/d);
}
//直线与圆的交点 (按照p1的远近顺序排列)
vP isCL(Point c,double r,Point p1,Point p2) 
{
  vP ret;
  ret.clear();
  double x=(p1-c)*(p2-p1);
  double y=(p2-p1).dis2();
  double d=x*x-y*((p1-c).dis2()-r*r);
  if(d<-Eps)return ret;
  if(d<0)d=0;
  Point q1=p1-((p2-p1)*(x/y));
  Point q2=p2-((p1)*(sqrt(d)/y));
  ret.push_back(q1);
  ret.push_back(q2);
  return ret;
}
//两圆的交点
vP isCC(Point c1,double r1,Point c2,double r2) 
{
  vP ret;
  ret.clear();
  double x=(c1-c2).dis2();
  double y=((r1*r1-r2*r2)/x+1)/2;
  double d=r1*r1/x-y*y;
  if(d<-Eps)return ret;
  if(d<0)d=0;
  Point q1=c1+((c2-c1)*y);
  Point q2=((c2-c1)*sqrt(d)).rot90();
  ret.push_back(q1);
  ret.push_back(q2);
  return ret;
}
//点P与圆的切点
vP tanCP(Point c,double r,Point p) 
{
  vP ret;ret.clear();
  double x=(p-c).dis2();
  double d=x-r*r;
  if(d<-Eps) return ret;
  if(d<0)d=0;
  Point q1=(p-c)*(r*r/x);
  Point q2=((p-c)*(-r*sqrt(d)/x)).rot90();
  ret.push_back(q1);
  ret.push_back(q2);
  return ret;
}

bool cmp_cH(const Point& a,const Point& b)
{
  if(a.x==b.x)return a.y<b.y;
  return a.x<b.x;
}

//凸包 逆时针
vP convexHull(vP ps)
{
  int n=ps.size(),k=0;
  if(n<=1)return ps;
  sort(ps.begin(),ps.end(),cmp_cH);
  vP qs;
  for(int i=0;i<n;qs[k++]=ps[i++])
    while(k>1 && (qs[k-1]-qs[k-1])%(ps[i]-qs[k-1])<Eps)k--;
  for(int i=n-2,t=k;i>=0;qs[k++]=ps[i--])
    while(k>t && (qs[k-1]-qs[k-2])%(ps[i]-qs[k-1])<Eps)k--;
  vP ret;
  for(int i=0;i<k-1;i++)
    ret.push_back(qs[i]);
  return ret;
}
  
//点在多边形内部判定 
//内部 1        边上 0       外部 -1
int contains(vP ps,Point q)
{
  int n=ps.size();
  int ret=-1;
  for(int i=0;i<n;i++)
  {
    Point a=ps[i]-q,b=ps[(i+1)%n]-q;
    if(a.y>b.y) swap(a,b);
    if(a.y<Eps && b.y>Eps && a%b>Eps) ret=-ret;
    if(abs(a%b)<Eps && a*b<Eps) return 0;
  }
  return ret;
}

vP convexCut(vP ps,Point p1,Point p2)
{
  int n=ps.size();
  vP ret;
  for(int i=0;i<n;i++)
  {
    int d1=sig((p2-p1)%(ps[i]-p1));
    int d2=sig((p2-p1)%(ps[(i+1)%n]-p1));
    if(d1>=0) ret.push_back(ps[i]);
    if(d1*d2<0)ret.push_back(isLL(p1,p2,ps[i],ps[(i+1)%n]));
  }
  return ret;
}

double convexDiameter(vP ps)
{
  int n=ps.size();
  int is=0,js=0;
  for(int i=1;i<n;i++)
  {
    if(ps[i].x>ps[is].x)is=i;
    if(ps[i].x<ps[is].x)js=i;
  }
  double maxd=(ps[is]-ps[js]).dis();
  int i=is,j=js;
  do
  {
    if((ps[(i+1)%n]-ps[i])%(ps[(j+1)%n]-ps[j])>=0)
      j=(j+1)%n;
    else 
      i=(i+1)%n;
    upd_max(maxd,(ps[i]-ps[j]).dis());
  } while(i!=is || j!=js);
  return maxd;
}

int main()
{
  return 0;
}

