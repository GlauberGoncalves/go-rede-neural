package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	rede "github.com/glaubergoncalves/go-rede-neural/rede"
)

func main() {

	alvo, entradas := extraiDadosDoDataset("./glaubergoncalves/go-rede-neural/exemplos/iris/Iris.csv")
	redeNeural := rede.NewRedeNeural(1, 3, 1)
	log.Println("treinando")
	treinaRede(redeNeural, entradas, alvo, 10000)
	log.Println("aprendizado concluido")

	resultado := redeNeural.Prever(entradas[0])
	fmt.Println("Deveria ser maior que 0.5 , Iris-setosa e foi : ", resultado)
}

func treinaRede(redeNeural rede.RedeNeural, entradas, alvo [][]float64, quantidade int) {
	for i := 0; i < quantidade; i++ {
		for i, dado := range entradas {
			redeNeural.Treinar(dado, alvo[i])
		}
	}
}

func extraiDadosDoDataset(path string) ([][]float64, [][]float64) {

	entradas := [][]float64{}
	alvo := [][]float64{}

	csvfile, err := os.Open(path)
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}
	r := csv.NewReader(csvfile)

	dados, _ := r.ReadAll()

	for _, dado := range dados {
		linhaAux := []float64{}
		for index, coluna := range dado {
			numero, err := strconv.ParseFloat(coluna, 64)
			if err != nil {
				log.Println("erro na conversão string para float64")
			}
			if index == 4 {
				if numero == 1 {
					alvo = append(alvo, []float64{1})
				} else {
					alvo = append(alvo, []float64{0})
				}

			} else {
				linhaAux = append(linhaAux, numero)
			}
		}
		entradas = append(entradas, linhaAux)
	}
	return alvo, entradas
}
