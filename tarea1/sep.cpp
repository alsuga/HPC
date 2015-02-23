#include <bits/stdc++.h>

using namespace std;

int main(){
  int n;
  double tmp;
  ofstream x ("x.in");
  ofstream ySec ("ySec.in");
  ofstream yPar ("yPar.in");
  ofstream xAc ("xAc.in");
  while(cin >> n) {
    x << n << endl;
    cin >> tmp;
    ySec << tmp << endl;
    cin >> tmp;
    yPar << tmp << endl;
    cin >> tmp;
    xAc << tmp << endl;
  }
  x.close();
  ySec.close();
  yPar.close();
  xAc.close();
  return 0;
}
