package matriz

import (
	"fmt"
	"math/rand"
	"time"
)

type Matriz struct {
	Linhas  int
	Colunas int
	Dados   [][]float64
}

func (m *Matriz) Iniciar(linhas, colunas int) {
	m.Colunas = colunas
	m.Linhas = linhas

	for i := 0; i < m.Linhas; i++ {
		arr := []float64{}
		for j := 0; j < m.Colunas; j++ {
			arr = append(arr, 0)
		}
		m.Dados = append(m.Dados, arr)
	}
}

// Soma : Soma duas matrizes
func MultiplicacaoEscalar(a Matriz, escalar float64) Matriz {

	matriz := Matriz{}
	matriz.Iniciar(a.Linhas, a.Colunas)

	for i := 0; i < a.Linhas; i++ {
		for j := 0; j < a.Colunas; j++ {
			matriz.Dados[i][j] = a.Dados[i][j] * escalar
		}
	}

	return matriz
}

func Mapeia(matriz Matriz, mapeamento func(matriz Matriz, linha, coluna int) float64) Matriz {
	novaMatriz := Matriz{}
	novaMatriz.Iniciar(matriz.Linhas, matriz.Colunas)

	for i := 0; i < matriz.Linhas; i++ {
		for j := 0; j < matriz.Colunas; j++ {
			novaMatriz.Dados[i][j] = mapeamento(matriz, i,j)
		}
	}

	return novaMatriz
}

// hadamard : hadamard
func Hadamard(a, b Matriz) Matriz {

	matriz := Matriz{}
	matriz.Iniciar(a.Linhas, b.Colunas)

	for i := 0; i < a.Linhas; i++ {
		for j := 0; j < a.Colunas; j++ {
			matriz.Dados[i][j] = a.Dados[i][j] * b.Dados[i][j]
		}
	}

	return matriz
}

// Soma : Soma duas matrizes
func Soma(a, b Matriz) Matriz {

	matriz := Matriz{}
	matriz.Iniciar(a.Linhas, b.Colunas)

	for i := 0; i < a.Linhas; i++ {
		for j := 0; j < a.Colunas; j++ {
			matriz.Dados[i][j] = a.Dados[i][j] + b.Dados[i][j]
		}
	}

	return matriz
}

// Subtracao : Subtrai uma matriz pela outra
func Subtracao(a, b Matriz) Matriz {

	matriz := Matriz{}
	matriz.Iniciar(a.Linhas, b.Colunas)

	for i := 0; i < a.Linhas; i++ {
		for j := 0; j < a.Colunas; j++ {
			matriz.Dados[i][j] = a.Dados[i][j] - b.Dados[i][j]
		}
	}

	return matriz
}

// Multiplicacao : Multiplica duas matrizes
func Multiplicacao(a, b Matriz) Matriz {

	matriz := Matriz{}
	matriz.Iniciar(a.Linhas, b.Colunas)

	for linha := 0; linha < a.Linhas; linha++ {
		for coluna := 0; coluna < b.Colunas; coluna++ {
			for i := 0; i < a.Colunas; i++ {
				matriz.Dados[linha][coluna] = matriz.Dados[linha][coluna] + (a.Dados[linha][i] * b.Dados[i][coluna])
			}
		}
	}

	return matriz
}

func (m *Matriz) MatrizTransposta() Matriz{
	matriz := Matriz{}
	matriz.Iniciar(m.Colunas, m.Linhas)

	for i := 0; i < m.Linhas; i++ {
		for j := 0; j < m.Colunas; j++ {
			matriz.Dados[j][i] = m.Dados[i][j]
		}
	}
	return matriz
}

func ArrayParaMatriz(array []float64) Matriz {
	m := Matriz{}
	m.Iniciar(len(array), 1)

	for i := 0; i < len(array); i++ {
		m.Dados[i][0] = array[i]
	}
	return m
}

func numeroAleatoreo() float64 {
	rand.Seed(time.Now().UTC().UnixNano())
	return rand.Float64()
}

func (m *Matriz) Randomizar(){

	for linha:=0; linha<m.Linhas; linha++{
		for coluna:=0; coluna<m.Colunas; coluna++{
			m.Dados[linha][coluna] = numeroAleatoreo()
		}
	}
}

func(m *Matriz) ImprimeMatriz() {

	for _, linha := range m.Dados {
		fmt.Print("|")
		for _, num := range linha {
			fmt.Print("\t", num)
		}
		fmt.Println("\t|")
	}
	fmt.Println("\n---------------")
}
