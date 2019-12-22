package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/glaubergoncalves/go-rede-neural/rede"
)

func main() {

	rand.Seed(time.Now().UTC().UnixNano())

	entradas := [][]float64{
		[]float64{1, 1},
		[]float64{1, 0},
		[]float64{0, 1},
		[]float64{0, 0},
	}
	alvo := [][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}

	redeNeural := rede.RedeNeural{}
	redeNeural.Inicia(2, 3, 1)

	for i := 0; i < 1000000; i++ {
		redeNeural.Treinar(entradas[0], alvo[0])
		redeNeural.Treinar(entradas[1], alvo[1])
		redeNeural.Treinar(entradas[2], alvo[2])
		redeNeural.Treinar(entradas[3], alvo[3])
	}

	resultado := redeNeural.Prever(entradas[0])
	fmt.Println("entrada: ", entradas[0], " resultado: ", resultado)

	resultado = redeNeural.Prever(entradas[1])
	fmt.Println("entrada: ", entradas[1], " resultado: ", resultado)

	resultado = redeNeural.Prever(entradas[2])
	fmt.Println("entrada: ", entradas[2], " resultado: ", resultado)

	resultado = redeNeural.Prever(entradas[3])
	fmt.Println("entrada: ", entradas[3], " resultado: ", resultado)

}
