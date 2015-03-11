#include <bits/stdc++.h>
using namespace std;
int main(){
  int n;
  double tmp;
  ofstream x ("x32.in");
  ofstream ySec ("ySec32.in");
  ofstream yPar ("yPar32.in");
  ofstream xAc ("xAc32.in");
  ofstream yTil ("yTil32.in");
  ofstream xTAc ("xTAc32.in");
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
