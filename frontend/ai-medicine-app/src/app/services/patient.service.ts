import { Injectable } from '@angular/core';
import axios from 'axios';

@Injectable({
  providedIn: 'root'
})
export class PatientService {

  private apiUrl = 'http://127.0.0.1:5000';  // Flask Backend URL

  constructor() {}

  addPatient(patient: any) {
    return axios.post(`${this.apiUrl}/add_patient`, patient, {
      headers: { 'Content-Type': 'application/json' },
      withCredentials: false
    });
  }

  getPrediction(features: any) {
    return axios.post(`${this.apiUrl}/predict`, { features }, {
      headers: { 'Content-Type': 'application/json' },
      withCredentials: false  // ✅ Fixes CORS issues
    });
  }
}
