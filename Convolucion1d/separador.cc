#include <bits/stdc++.h>
using namespace std;
int main(){
  int n;
  double tmp;
  string no;
  ofstream x ("x.in");
  ofstream ySec ("ySec.in");
  ofstream yParB ("yParB.in");
  ofstream yParC ("yParC.in");
  ofstream yParT ("yParT.in");
  ofstream yAcB ("yAcB.in");
  ofstream yAcC ("yAcC.in");
  ofstream yAcT ("yAcT.in");
  while(cin >> n) {
    x << n << endl;

    cin >> tmp;
    ySec << tmp << endl;

    cin >> tmp;
    yParB << tmp << endl;

    cin >> tmp;
    yAcB << tmp << endl;

    cin >> tmp;
    yParC << tmp << endl;

    cin >> tmp;
    yAcC << tmp << endl;

    cin >> tmp;
    yParT << tmp << endl;

    cin >> tmp;
    yAcT << tmp << endl;
  }
  x.close();
  ySec.close();
  yParB.close();
  yAcB.close();
  yParC.close();
  yAcC.close();
  yParT.close();
  yAcT.close();
  return 0;
}
