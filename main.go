package main

import (
	"github.com/go-martini/martini"
	"net/http"
	"log"
	"encoding/json"
	"os"
)

func main() {
	os.Setenv("PORT","8000")
	m := martini.Classic()
	m.Post("/image", func(res http.ResponseWriter, req *http.Request, log *log.Logger) {
		req.ParseForm()
		data := req.Form.Get("img")
		d := predict(data)
		write, err := json.Marshal(d)
		if err != nil {
			res.WriteHeader(500)
			log.Fatal(err)
		}
		res.WriteHeader(200)
		res.Write(write)
	})
	m.Run()
}
