/**
 * Next.js Proxy (formerly middleware.ts) — Item #4: Force HTTPS in production.
 * Renamed from middleware.ts → proxy.ts per Next.js 16 deprecation.
 * Runs at the edge before every request.
 */
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function proxy(request: NextRequest) {
  // Force HTTPS in production
  if (process.env.NODE_ENV === "production") {
    const url = request.nextUrl;
    if (url.protocol === "http:") {
      url.protocol = "https:";
      return NextResponse.redirect(url, { status: 301 });
    }
  }

  const response = NextResponse.next();

  // HSTS header (Item #4)
  if (process.env.NODE_ENV === "production") {
    response.headers.set(
      "Strict-Transport-Security",
      "max-age=31536000; includeSubDomains; preload"
    );
  }

  return response;
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
