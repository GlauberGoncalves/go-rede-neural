package rede

import (
	"math"

	"github.com/glaubergoncalves/go-rede-neural/matriz"
)

type RedeNeural struct {
	entrada               int
	escondida             int
	Saida                 int
	biasEntradaParaOculta matriz.Matriz
	biasOcultaParaSaida   matriz.Matriz
	pesoEntradaEscondida  matriz.Matriz
	pesoEscondidaSaida    matriz.Matriz
	TaxaAprendizado       float64
}

func NewRedeNeural(entradas, ocultas, saidas int) RedeNeural {
	r := RedeNeural{}
	r.Inicia(entradas, ocultas, saidas)
	return r
}

func sigmoidDerivada(x float64) float64 {
	return x * (1 - x)
}

func (r *RedeNeural) Inicia(entrada, escondida, saida int) {

	r.entrada = entrada
	r.escondida = escondida
	r.Saida = saida

	r.biasEntradaParaOculta = matriz.Matriz{}
	r.biasOcultaParaSaida = matriz.Matriz{}
	r.pesoEntradaEscondida = matriz.Matriz{}
	r.pesoEntradaEscondida = matriz.Matriz{}

	r.biasEntradaParaOculta.Iniciar(escondida, 1)
	r.biasEntradaParaOculta.Randomizar()
	r.biasOcultaParaSaida.Iniciar(saida, 1)
	r.biasOcultaParaSaida.Randomizar()

	r.pesoEntradaEscondida.Iniciar(escondida, entrada)
	r.pesoEntradaEscondida.Randomizar()
	r.pesoEscondidaSaida.Iniciar(saida, escondida)
	r.pesoEscondidaSaida.Randomizar()

	r.TaxaAprendizado = 0.1
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Ativacao(matriz matriz.Matriz) {
	for i := 0; i < matriz.Linhas; i++ {
		for j := 0; j < matriz.Colunas; j++ {
			matriz.Dados[i][j] = sigmoid(matriz.Dados[i][j])
		}
	}
}

func (r *RedeNeural) feedForward(_entrada []float64) matriz.Matriz {

	// camada de entrapa para a escondida
	entrada := matriz.ArrayParaMatriz(_entrada)
	escondida := matriz.Multiplicacao(r.pesoEntradaEscondida, entrada)
	escondida = matriz.Soma(escondida, r.biasEntradaParaOculta)
	Ativacao(escondida)

	// camada de escondida para saida

	saida := matriz.Multiplicacao(r.pesoEscondidaSaida, escondida)
	saida = matriz.Soma(saida, r.biasOcultaParaSaida)

	Ativacao(saida)
	return saida
}

func (r *RedeNeural) Treinar(arrayEntrada []float64, alvo []float64) {

	// camada de entrapa para a escondida
	entrada := matriz.ArrayParaMatriz(arrayEntrada)
	escondida := matriz.Multiplicacao(r.pesoEntradaEscondida, entrada)
	escondida = matriz.Soma(escondida, r.biasEntradaParaOculta)
	Ativacao(escondida)

	// camada de escondida para saida

	saida := matriz.Multiplicacao(r.pesoEscondidaSaida, escondida)
	saida = matriz.Soma(saida, r.biasOcultaParaSaida)
	Ativacao(saida)

	// backpropagation

	// saida --> oculta
	valorEsperado := matriz.ArrayParaMatriz(alvo)
	erroSaida := matriz.Subtracao(valorEsperado, saida)
	derivadaSaida := matriz.Mapeia(saida, func(saida matriz.Matriz, linha, coluna int) float64 {
		return sigmoidDerivada(saida.Dados[linha][coluna])
	})
	escondidaTransposta := escondida.MatrizTransposta()

	gradiente := matriz.Hadamard(derivadaSaida, erroSaida)
	gradiente = matriz.MultiplicacaoEscalar(gradiente, r.TaxaAprendizado)

	r.biasOcultaParaSaida = matriz.Soma(r.biasOcultaParaSaida, gradiente)
	pesoEscondidaSaidaDelta := matriz.Multiplicacao(gradiente, escondidaTransposta)
	r.pesoEscondidaSaida = matriz.Soma(r.pesoEscondidaSaida, pesoEscondidaSaidaDelta)

	pesoEscondidaSaidaTransposto := r.pesoEscondidaSaida.MatrizTransposta()
	erroEscondida := matriz.Multiplicacao(pesoEscondidaSaidaTransposto, erroSaida)
	derivadaEscondida := matriz.Mapeia(escondida, func(escondida matriz.Matriz, linha, coluna int) float64 {
		return sigmoidDerivada(escondida.Dados[linha][coluna])
	})
	entradaTransposta := entrada.MatrizTransposta()
	gradienteEscondida := matriz.Hadamard(derivadaEscondida, erroEscondida)
	gradienteEscondida = matriz.MultiplicacaoEscalar(gradienteEscondida, r.TaxaAprendizado)

	// Adjust Bias O->H
	r.biasEntradaParaOculta = matriz.Soma(r.biasEntradaParaOculta, gradienteEscondida)

	pesoEntradaEscondidaDelta := matriz.Multiplicacao(gradienteEscondida, entradaTransposta)
	r.pesoEntradaEscondida = matriz.Soma(r.pesoEntradaEscondida, pesoEntradaEscondidaDelta)

}

func (r *RedeNeural) Prever(arrayEntrada []float64) [][]float64 {
	// entrada -> escondida
	entrada := matriz.ArrayParaMatriz(arrayEntrada)

	escondida := matriz.Multiplicacao(r.pesoEntradaEscondida, entrada)
	escondida = matriz.Soma(escondida, r.biasEntradaParaOculta)

	escondida = matriz.Mapeia(escondida, func(saida matriz.Matriz, linha, coluna int) float64 {
		return sigmoid(escondida.Dados[linha][coluna])
	})

	// escondida -> saida
	saida := matriz.Multiplicacao(r.pesoEscondidaSaida, escondida)
	saida = matriz.Soma(saida, r.biasOcultaParaSaida)
	saida = matriz.Mapeia(saida, func(saida matriz.Matriz, linha, coluna int) float64 {
		return sigmoid(saida.Dados[linha][coluna])
	})
	arraySaida := saida.Dados

	return arraySaida
}
