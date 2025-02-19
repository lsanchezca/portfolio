package es.urjc.grafo.ABII.Algorithms;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Export {

    public static void exportToCSV(List<String> routeDetails, String instanceName) {
        try (FileWriter writer = new FileWriter("optimization_results_" + instanceName + ".csv")) {
            // Escribir cabecera
            writer.append("Vehicle, Route\n");

            // Escribir los detalles de la ruta
            for (String route : routeDetails) {
                writer.append(route + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}