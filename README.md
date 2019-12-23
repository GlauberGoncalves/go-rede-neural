# go-rede-neural

Rede neural do zero utilizando matrizes para realizar o feedforward e backpropagation


Instalação
```sh
go get -u github.com/GlauberGoncalves/go-rede-neural/rede
```
A rede utiliza 3 camadas: entrada, oculta e saida
Informe a entrada e a quantidade de neuronios em cada camada

```golang
package main

import (
	rede "github.com/glaubergoncalves/go-rede-neural/rede"
)

redeNeural := rede.NewRedeNeural(2, 3, 1)
```

Treinando a rede

```golang
redeNeural.Treinar(entrada, alvos)
```

Realizando previsões

```golang
resultado := redeNeural.Prever(entradas)
fmt.Println("resultado deveria ser proximo a 0 e foi ", resultado)
```

