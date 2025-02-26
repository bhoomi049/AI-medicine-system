import { Component } from '@angular/core';
import { PatientService } from './services/patient.service';
import { CommonModule } from '@angular/common'; // For *ngIf
import { FormsModule } from '@angular/forms';   // For [(ngModel)]

@Component({
  selector: 'app-root',
  standalone: true, // ✅ Standalone component
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  imports: [CommonModule, FormsModule], // ✅ Add necessary modules
})
export class AppComponent {
  title = 'AI Medicine System';

  // Use 'any' or a custom interface instead of 'string'
  patient = { name: '', age: 0, disease: '' };
  treatment: any = null; // ✅ Changed from '' to any = null
  isLoading = false;     // For *ngIf loading spinner

  constructor(private patientService: PatientService) {}

  addPatient() {
    this.isLoading = true;
    this.patientService.addPatient(this.patient)
      .then(response => {
        alert(response.data.message);
        this.isLoading = false;
      })
      .catch(error => {
        console.log(error);
        this.isLoading = false;
      });
  }

  predictTreatment() {
    this.isLoading = true;
    // Send an object with a 'symptoms' array (your backend expects this)
    const features = { symptoms: [this.patient.disease] };

    this.patientService.getPrediction(features)
      .then(response => {
        // The backend returns { treatment: {...} }
        this.treatment = response.data.treatment;
        this.isLoading = false;
      })
      .catch(error => {
        console.log(error);
        this.isLoading = false;
      });
  }
}
