package main

import (
	"fmt"

	"github.com/glaubergoncalves/go-rede-neural/rede"
)

func main() {

	// XOR Problem
	entradas := [][]float64{
		{1, 1},
		{1, 0},
		{0, 1},
		{0, 0},
	}

	alvos := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	redeNeural := rede.NewRedeNeural(2, 3, 1)

	// treinando a rede
	for i := 0; i < 10000; i++ {
		for index, entrada := range entradas {
			redeNeural.Treinar(entrada, alvos[index])
		}
	}

	// tendando prever resultado

	resultado := redeNeural.Prever(entradas[0])
	fmt.Println("resultado deveria ser proximo a 0 e foi ", resultado)

	resultado = redeNeural.Prever(entradas[1])
	fmt.Println("resultado deveria ser proximo a 1 e foi ", resultado)

}
