Convolución de imagenes
=======================

Implementación
--------------

Fueron implementadas 2 versiones de paso a grises, secuencial y paralela aplicando el filtro en X y en Y, ademas se implementaron 4 versiones de la convolución en 2d, secuencial, paralela con memoria global, paralela con caching y paralela con tiling, pero esta ultima tiene un error donde se pierden ciertos pixeles, por lo tanto se omitieron los datos de esta.

Tablas y gráficas
-----------------

![](https://github.com/alsuga/HPC/blob/master/Convolucion2d/tabla.png "Tabla")

![](https://github.com/alsuga/HPC/blob/master/Convolucion2d/cpuvsgpu.png "CPU vs GPU")


![](https://github.com/alsuga/HPC/blob/master/Convolucion2d/aceleracion.png "Aceleracion")



Conclusiones
------------

Dados los datos tomados de las pruebas se puede concluir que:

* Es muy notable que el algoritmo secuencial es mucho mas lento que la
implementación paralela, tanto de cambio a grises como la convolusión
en si cuando supera los 600x600 pixeles.

* Aunque la implementación de tiling no funcionaba correctamente
se noto que el algoritmo con caching tenia tiempos muy parecidos
y a su vez estos tenian aceleraciones más altas que la memoria gobal,
en este caso hasta 4x.

* Se noto que en imagenes pequeñas, la implementación con memoria global
  daba aceleraciones muy parecidas a las de la tecnica de caching, con
una diferencia cercana a 10^-3X.
