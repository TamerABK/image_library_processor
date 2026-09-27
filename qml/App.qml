pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import "theme" as Room36Theme
import "pages" as Pages
import "dev" as Dev

ApplicationWindow {
    id: window

    property var appBridge: null
    property string startPage: "home"
    property bool useHomePreviewData: false
    readonly property string currentPage: normalizePage(startPage)

    visible: true
    width: 1440
    height: 900
    minimumWidth: metrics.minimumWindowWidth
    minimumHeight: metrics.minimumWindowHeight
    title: "Room 36"
    color: pageBackground(currentPage)

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }

    function normalizePage(pageId) {
        if (pageId === "loading" || pageId === "auth" || pageId === "home" || pageId === "gallery") {
            return pageId
        }
        return "home"
    }

    function pageBackground(pageId) {
        if (pageId === "auth") {
            return theme.authBackground
        }
        if (pageId === "gallery") {
            return theme.galleryBackground
        }
        if (pageId === "loading") {
            return theme.mainBlue
        }
        return theme.homeBackground
    }

    function showPage(pageId) {
        startPage = normalizePage(pageId)
    }

    Loader {
        id: pageLoader
        objectName: "pageLoader"
        anchors.fill: parent
        sourceComponent: {
            if (window.currentPage === "loading") {
                return loadingPageComponent
            }
            if (window.currentPage === "auth") {
                return authPageComponent
            }
            if (window.currentPage === "gallery") {
                return galleryComponent
            }
            if (window.useHomePreviewData) {
                return homePreviewComponent
            }
            return homePageComponent
        }
    }

    Component {
        id: loadingPageComponent

        Pages.LoadingPage {
            objectName: "loadingPage"
            onContinueRequested: window.showPage("home")
        }
    }

    Component {
        id: authPageComponent

        Pages.AuthPage {
            objectName: "authPage"
        }
    }

    Component {
        id: homePageComponent

        Pages.HomePage {
            objectName: "homePage"
        }
    }

    Component {
        id: homePreviewComponent

        Dev.ScreenPreview {
            objectName: "screenPreview"
        }
    }

    Component {
        id: galleryComponent

        Dev.ComponentGallery {
            objectName: "componentGallery"
            bridge: window.appBridge
        }
    }
}
