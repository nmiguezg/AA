El script deteccion_barcos.jl se puede ejecutar por línea de comandos de la siguiente forma:

	$ julia deteccion_barcos.jl -model [RNA|ANN|SVM|DT] -image [IMAGE_PATH] [-output [OUTPUT_IMAGE_PATH]]"

O bien dándole permisos de ejecucción:

	$ chmod +x d deteccion_barcos.jl
	$ ./deteccion_barcos.jl -model [RNA|ANN|SVM|DT] -image [IMAGE_PATH] [-output [OUTPUT_IMAGE_PATH]]"
 
 
Las imágenes a probar deben estar en formato .png o .jpeg, se adjunta el archivo barcos_sin_recortar.zip con la totalidad de las imágenes de la bbdd descargada en la cual aún están los barcos con mar sin recortar, con las que se puede probar la detección. El script tarda normalmente alrededor de 50s en ejecutarse con cada imagen. Descomprimido el zip en la carpeta barcos_sin_recortar un ejemplo de ejecución con el modelo KNN sería:


	$ julia deteccion_barcos.jl -model KNN -image barcos_sin_recortar/boat12.png -output detected_boat12.png"
