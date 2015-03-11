#include <bits/stdc++.h>
using namespace std;
int main(){
  int n;
  double tmp;
  ofstream x ("x4.in");
  ofstream ySec ("ySec4.in");
  ofstream yPar ("yPar4.in");
  ofstream xAc ("xAc4.in");
  ofstream yTil ("yTil4.in");
  ofstream xTAc ("xTAc4.in");
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
