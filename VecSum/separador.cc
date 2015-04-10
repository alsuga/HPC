#include <bits/stdc++.h>
using namespace std;
int main(){
  int n;
  double tmp;
  string no;
  ofstream x ("x32F.in");
  ofstream ySec ("ySec32F.in");
  ofstream yPar ("yPar32F.in");
  ofstream yAc ("xAc32F.in");
  while(cin >> n) {
    x << n << endl;
    cin >> tmp;
    ySec << tmp << endl;
    cin >> tmp;
    yPar << tmp << endl;
    cin >> tmp;
    yAc << tmp << endl;
  }
  x.close();
  ySec.close();
  yPar.close();
  yAc.close();
  return 0;
}
