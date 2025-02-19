package es.urjc.grafo.ABII.Algorithms;

import es.urjc.grafo.ABII.Model.Instance;
import es.urjc.grafo.ABII.Model.Evaluator;
import es.urjc.grafo.ABII.Model.Solution;

import java.util.*;

public class Algorithm1 implements Algorithm {

    static class VehicleState {
        double accumulatedDemand;
        double batteryAvailable;

        VehicleState(double accumulatedDemand, double batteryAvailable) {
            this.accumulatedDemand = accumulatedDemand;
            this.batteryAvailable = batteryAvailable;
        }
    }

    // ACO
    public Solution run(Instance instance) {
        long startTime = System.currentTimeMillis();
        int nVehicles = instance.getNumberOfVehicles();
        int maxIteraciones = 1000;
        double alpha = 1;
        double beta = 2;
        double depositedPheromones = 1 + Math.random() * 9;
        int numNodos = instance.getNumberOfCustomers() + instance.getNumberOfChargeStations() - 1;
        int threshold = 10;

        double[][] pheromones = inicializarFeromonas(numNodos);
        double NbestSol = Double.MAX_VALUE;
        Solution bestSol = null;


        for (int iter = 0; iter < maxIteraciones; iter++) {
            Set<Integer> visitedNodes = new HashSet<>();

            List<List<Integer>> rutasVehiculos = new ArrayList<>();
            for (int vehiculo = 0; vehiculo < nVehicles; vehiculo++) {
                rutasVehiculos.add(new ArrayList<>());
            }

            int nodosRestantes = instance.getNumberOfCustomers();
            int vehiculoActual = 0;
            boolean[] visitado = new boolean[numNodos];

            while (nodosRestantes > 0 && vehiculoActual < nVehicles) {
                double totalCapacity = instance.getCarryingCapacity();
                double batteryCapacity = instance.getBatteryCapacity();
                VehicleState state = new VehicleState(0.0, batteryCapacity);
                List<Integer> route = new ArrayList<>();

                int actual = 0;
                route.add(actual);
                visitedNodes.add(actual);

                while (nodosRestantes > 0) {
                    int nextNode = addNode(pheromones, actual, alpha, beta, visitado, instance, state, totalCapacity, batteryCapacity, threshold, visitedNodes);

                    if (nextNode == -1) {
                        break;
                    } else {
                        route.add(nextNode);
                        actual = nextNode;
                        if (!(instance.isChargeStation(nextNode + 1)) && nextNode != 0) {
                            nodosRestantes--;
                            visitado[nextNode] = true;
                        }
                    }
                }
                if (state.batteryAvailable >= instance.getDistance(actual + 1, 1) * instance.getH()) {
                    route.add(0);
                } else {
                    int estacionCargaMasCercana = -1;
                    double distanciaMinima = Double.MAX_VALUE;
                    for (int i = 0; i < pheromones.length; i++) {
                        if (instance.isChargeStation(i + 1) || i == 0) {
                            double distanciaAEstacion = instance.getDistance(actual + 1, i + 1);
                            if (distanciaAEstacion < distanciaMinima) {
                                distanciaMinima = distanciaAEstacion;
                                estacionCargaMasCercana = i;
                            }
                        }
                    }

                    if (estacionCargaMasCercana != -1) {
                        state.batteryAvailable = batteryCapacity;
                        route.add(estacionCargaMasCercana);
                        route.add(0);
                    }
                }

                rutasVehiculos.set(vehiculoActual, route);

                vehiculoActual++;
            }


            List<List<Integer>> rutasVehiculosIncrementadas = new ArrayList<>();
            for (List<Integer> ruta : rutasVehiculos) {
                List<Integer> rutaIncrementada = new ArrayList<>();
                for (int nodo : ruta) {
                    rutaIncrementada.add(nodo + 1);
                }
                rutasVehiculosIncrementadas.add(rutaIncrementada);
            }

            List<Integer>[] rutasArray = rutasVehiculosIncrementadas.toArray(new List[0]);
            Solution solution = new Solution(rutasArray);

            if (solution != null && (Evaluator.isFeasible(solution, instance))) {
                double score = Evaluator.evaluate(solution, instance);

                if (score < NbestSol) {
                    NbestSol = score;
                    bestSol = solution;
                }

                double p = 0.1;
                updatePheromones(pheromones, rutasVehiculos, p, depositedPheromones, score);

            }
        }

        imprimirSol(NbestSol, bestSol);
        long endTime = System.currentTimeMillis();
        long durationInMillis = endTime - startTime;
        double durationInSeconds = durationInMillis / 1000.0;
        return bestSol;
    }

    public void imprimirSol(double score, Solution solution) {
        List<Integer>[] rutas = solution.routes();
        String solucionFormateada = obtenerSolucionFormateada(rutas, score);
        System.out.println(score + " " + solucionFormateada);
    }

    public String obtenerSolucionFormateada(List<Integer>[] rutas, double score) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rutas.length; i++) {
            List<Integer> ruta = rutas[i];
            sb.append("[");
            for (int j = 0; j < ruta.size(); j++) {
                int nodo = ruta.get(j);
                sb.append(nodo);
                if (j < ruta.size() - 1) {
                    sb.append(",");
                }
            }
            sb.append("]");
            if (i < rutas.length - 1) {
                sb.append(";");
            }
        }

        return sb.toString();
    }

    private double[][] inicializarFeromonas(int numNodos) {
        double[][] pheromones = new double[numNodos][numNodos];
        Random random = new Random();

        for (int i = 0; i < numNodos; i++) {
            for (int j = 0; j < numNodos; j++) {
                pheromones[i][j] = random.nextDouble() / 1000;
            }
        }
        return pheromones;
    }


    private int addNode(
            double[][] pheromones,
            int actual,
            double alpha,
            double beta,
            boolean[] visitado,
            Instance instance,
            VehicleState state,
            double totalCapacity,
            double batteryCapacity,
            double threshold,
            Set<Integer> visitedNodes) {
        double[] probabilidades = new double[pheromones.length];
        double suma = 0.0;
        boolean capacity = false;
        boolean battery = false;

        for (int i = 0; i < pheromones.length; i++) {
            if (!visitado[i] && !instance.isChargeStation(i + 1) && i != 0) {
                double demanda = instance.getDemand(i + 1);
                double distancia = instance.getDistance(actual + 1, i + 1);
                if (distancia == 0.0) distancia = 1e-6;
                double carga = distancia * instance.getH();

                boolean cumpleCapacidad = (state.accumulatedDemand + demanda <= totalCapacity);
                boolean cumpleBateria = (state.batteryAvailable >= carga);


                boolean puedeLlegarEstacion = false;
                if (ultimoNodo(actual, visitado, instance)){
                    puedeLlegarEstacion = true;
                    battery = true;
                }
                else{
                    for (int j = 0; j < pheromones.length; j++) {
                        if (instance.isChargeStation(j + 1) || j == 0) {
                            double distanciaAEstacion = instance.getDistance(i + 1, j + 1);
                            double cargaAEstacion = distanciaAEstacion * instance.getH();
                            if (state.batteryAvailable - carga >= cargaAEstacion) {
                                puedeLlegarEstacion = true;
                                battery = true;
                                break;
                            }
                        }
                    }
                }

                if (cumpleCapacidad && cumpleBateria && puedeLlegarEstacion) {
                    double tau = Math.pow(pheromones[actual][i], alpha);
                    double eta = Math.pow(1.0 / distancia, beta);
                    eta = Math.max(eta, 1e-6);

                    probabilidades[i] = tau * eta;
                    suma += probabilidades[i];
                } else {
                    probabilidades[i] = 0.0;
                }
            } else {
                probabilidades[i] = 0.0;
            }
        }

        if ((visitedNodes.size() == instance.getNumberOfCustomers()) ||(instance.isChargeStation(actual +1))){
            battery = true;
        }

        if ((suma == 0.0) && (visitedNodes.size() <= instance.getNumberOfCustomers()-1) && (!(battery))){
            int estacionCargaMasCercana = -1;
            double distanciaMinima = Double.MAX_VALUE;
            for (int i = 0; i < pheromones.length; i++) {
                if (instance.isChargeStation(i + 1) || i == 0) {
                    double distanciaAEstacion = instance.getDistance(actual + 1, i + 1);
                    if (distanciaAEstacion < distanciaMinima) {
                        distanciaMinima = distanciaAEstacion;
                        estacionCargaMasCercana = i;
                    }
                }
            }



            if (estacionCargaMasCercana != -1) {
                state.batteryAvailable = batteryCapacity;
                return estacionCargaMasCercana;
            }
        }

        if (suma == 0.0) return -1;

        double r = Math.random() * suma;
        double acumulado = 0.0;
        for (int i = 0; i < probabilidades.length; i++) {
            acumulado += probabilidades[i];
            if (acumulado >= r) {
                state.accumulatedDemand += instance.getDemand(i + 1);
                double distanciaSeleccionada = instance.getDistance(actual + 1, i + 1);
                double cargaSeleccionada = distanciaSeleccionada * instance.getH();
                state.batteryAvailable -= cargaSeleccionada;

                visitedNodes.add(i);
                return i;
            }
        }

        return -1;
    }

    private boolean ultimoNodo(int actual, boolean[] visitado, Instance instance) {
        int countNodosNoVisitados = 0;
        for (int i = 0; i < visitado.length; i++) {
            if ((!(visitado[i])) && (!(instance.isChargeStation(actual + 1) && actual != 0))) {
                countNodosNoVisitados ++;
            }
        }
        if (countNodosNoVisitados ==1){
            return true;
        }
        return false;
    }

    private void updatePheromones(double[][] pheromones, List<List<Integer>> rutasVehiculos, double p, double Q, double L) {
        int numNodos = pheromones.length;


        for (int i = 0; i < numNodos; i++) {
            for (int j = 0; j < numNodos; j++) {
                pheromones[i][j] *= (1 - p);
            }
        }

        for (List<Integer> ruta : rutasVehiculos) {
            for (int i = 0; i < ruta.size() - 1; i++) {
                int desde = ruta.get(i);
                int hasta = ruta.get(i + 1);

                double incremento = Q / L;
                pheromones[desde][hasta] += incremento;
                pheromones[hasta][desde] += incremento;
            }
        }
    }


    public String toString() {
        return "Algorithm1";
    }
}