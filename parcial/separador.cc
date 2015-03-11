#include <bits/stdc++.h>
using namespace std;
int main(){
  int n;
  double tmp;
  ofstream x ("x32F.in");
  ofstream ySec ("ySec32F.in");
  ofstream yPar ("yPar32F.in");
  ofstream xAc ("xAc32F.in");
  ofstream yTil ("yTil32F.in");
  ofstream xTAc ("xTAc32F.in");
  while(cin >> n) {
    x << n << endl;
    cin >> tmp;
    ySec << tmp << endl;
    cin >> tmp;
    yPar << tmp << endl;
    cin >> tmp;
    xAc << tmp << endl;
    cin >> tmp;
    yTil << tmp << endl;
    cin >> tmp;
    xTAc << tmp << endl;
  }
  x.close();
  ySec.close();
  yPar.close();
  yTil.close();
  xAc.close();
  xTAc.close();
  return 0;
}
