import { ApplicationConfig, importProvidersFrom } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { routes } from './app.routes';
import { CommonModule } from '@angular/common'; 
import { FormsModule } from '@angular/forms';  // ✅ Correct import

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    provideHttpClient(),
    importProvidersFrom(CommonModule, FormsModule) // ✅ Fix ngModel and *ngIf errors
  ]
};


