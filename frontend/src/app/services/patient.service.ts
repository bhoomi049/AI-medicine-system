import { Injectable } from '@angular/core';
import axios from 'axios';

@Injectable({
  providedIn: 'root'
})
export class PatientService {
  baseUrl = "http://127.0.0.1:5000";  // Flask API URL

  async addPatient(patientData: any) {
    return await axios.post(`${this.baseUrl}/add_patient`, patientData);
  }

  async getPrediction(features: any) {
    return await axios.post(`${this.baseUrl}/predict`, { features });
  }
}
